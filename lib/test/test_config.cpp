#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#include "config.hpp"

namespace {

static std::string fixture(const std::string& name) {
  return std::string(TEST_DATA_DIR) + "/" + name;
}

// ---------------------------------------------------------------------------
// TC1: Simple extends
// Template "cam_sg" (2 nodes). Instance "cam0" extends it.
// Expected: 2 nodes with prefixed names, local input reference also prefixed.
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC1_SimpleExtends) {
  stargazer::configuration cfg(fixture("test_simple_extends.json"));
  auto nodes = cfg.get_nodes("pipeline");

  ASSERT_EQ(nodes.size(), 2u);

  auto cam0_source = std::find_if(nodes.begin(), nodes.end(),
                                  [](const auto& n) { return n.name == "cam0_source"; });
  auto cam0_decoder = std::find_if(nodes.begin(), nodes.end(),
                                   [](const auto& n) { return n.name == "cam0_decoder"; });

  ASSERT_NE(cam0_source, nodes.end()) << "Expected node 'cam0_source' not found";
  ASSERT_NE(cam0_decoder, nodes.end()) << "Expected node 'cam0_decoder' not found";

  ASSERT_EQ(cam0_source->subgraph_instance, "cam0");
  ASSERT_EQ(cam0_decoder->subgraph_instance, "cam0");

  // Local input reference "source" must be rewritten to "cam0_source"
  ASSERT_TRUE(cam0_decoder->inputs.count("default") > 0);
  EXPECT_EQ(cam0_decoder->inputs.at("default"), "cam0_source");
}

// ---------------------------------------------------------------------------
// TC2: Direct nodes (no extends)
// Pipeline subgraph has inline nodes. Names must not be prefixed.
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC2_DirectNodes) {
  stargazer::configuration cfg(fixture("test_direct_nodes.json"));
  auto nodes = cfg.get_nodes("pipeline");

  ASSERT_EQ(nodes.size(), 2u);

  auto source = std::find_if(nodes.begin(), nodes.end(),
                             [](const auto& n) { return n.name == "source_node"; });
  auto sink =
      std::find_if(nodes.begin(), nodes.end(), [](const auto& n) { return n.name == "sink_node"; });

  ASSERT_NE(source, nodes.end()) << "Expected node 'source_node' not found";
  ASSERT_NE(sink, nodes.end()) << "Expected node 'sink_node' not found";

  // Input reference must be kept as-is (no extends → no prefix logic)
  ASSERT_TRUE(sink->inputs.count("default") > 0);
  EXPECT_EQ(sink->inputs.at("default"), "source_node");
}

// ---------------------------------------------------------------------------
// TC3: Parameter hierarchy
// Template params < instance params < node params.
// Template: fps=30, db_path="template_db"
// Instance:           db_path="cam0_db"
// Node:     fps=60
// Expected: node "cam0_loader" has fps=60, db_path="cam0_db"
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC3_ParamHierarchy) {
  stargazer::configuration cfg(fixture("test_param_hierarchy.json"));
  auto nodes = cfg.get_nodes("pipeline");

  ASSERT_EQ(nodes.size(), 1u);
  EXPECT_EQ(nodes[0].name, "cam0_loader");

  // Node-level fps (60) overrides template fps (30)
  EXPECT_EQ(nodes[0].get_param<std::int64_t>("fps"), 60);
  // Instance db_path overrides template db_path
  EXPECT_EQ(nodes[0].get_param<std::string>("db_path"), "cam0_db");
}

// ---------------------------------------------------------------------------
// TC4: Instance node override
// Instance defines nodes[] to override template node params.
// Template loader node has db_path="template_db".
// Instance overrides it with db_path="override_db".
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC4_NodeOverride) {
  stargazer::configuration cfg(fixture("test_node_override.json"));
  auto nodes = cfg.get_nodes("pipeline");

  ASSERT_EQ(nodes.size(), 1u);
  EXPECT_EQ(nodes[0].name, "cam0_loader");

  // Override should take effect over template value
  EXPECT_EQ(nodes[0].get_param<std::string>("db_path"), "override_db");
}

// ---------------------------------------------------------------------------
// TC5: Cross-subgraph input reference (dot-to-underscore conversion)
// Instance override sets inputs like "ext_sg.source_node".
// get_nodes() must convert '.' to '_'.
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC5_CrossRefDotToUnderscore) {
  stargazer::configuration cfg(fixture("test_cross_ref.json"));
  auto nodes = cfg.get_nodes("pipeline");

  // Find sg1_processor (prefixed because it extends consumer_template)
  auto proc = std::find_if(nodes.begin(), nodes.end(),
                           [](const auto& n) { return n.name == "sg1_processor"; });
  ASSERT_NE(proc, nodes.end()) << "Expected node 'sg1_processor' not found";

  ASSERT_TRUE(proc->inputs.count("cam0") > 0);
  ASSERT_TRUE(proc->inputs.count("cam1") > 0);

  // '.' in "ext_sg.source_node" must become '_'
  EXPECT_EQ(proc->inputs.at("cam0"), "ext_sg_source_node");
  EXPECT_EQ(proc->inputs.at("cam1"), "ext_sg_other_node");
}

// ---------------------------------------------------------------------------
// TC6: Multiple subgraphs (extends + direct)
// cam0 extends camera_template (2 nodes, prefixed).
// reconstructor has a direct node (no prefix).
// Total: 3 nodes.
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC6_MultipleSubgraphs) {
  stargazer::configuration cfg(fixture("test_multiple_subgraphs.json"));
  auto nodes = cfg.get_nodes("pipeline");

  ASSERT_EQ(nodes.size(), 3u);

  auto cam0_loader = std::find_if(nodes.begin(), nodes.end(),
                                  [](const auto& n) { return n.name == "cam0_loader"; });
  auto cam0_decoder = std::find_if(nodes.begin(), nodes.end(),
                                   [](const auto& n) { return n.name == "cam0_decoder"; });
  auto reconstruct = std::find_if(nodes.begin(), nodes.end(),
                                  [](const auto& n) { return n.name == "reconstruct_node"; });

  ASSERT_NE(cam0_loader, nodes.end()) << "Expected 'cam0_loader' not found";
  ASSERT_NE(cam0_decoder, nodes.end()) << "Expected 'cam0_decoder' not found";
  ASSERT_NE(reconstruct, nodes.end()) << "Expected 'reconstruct_node' not found";

  // cam0_decoder's local input must be prefixed
  EXPECT_EQ(cam0_decoder->inputs.at("default"), "cam0_loader");

  // reconstructor's input must be preserved as-is (cross-subgraph, no dot)
  EXPECT_EQ(reconstruct->inputs.at("cam0"), "cam0_decoder");
}

// ---------------------------------------------------------------------------
// TC7: Roundtrip (load → update() → reload → same get_nodes() result)
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC7_Roundtrip) {
  const std::string orig_path = fixture("test_roundtrip.json");
  const std::string temp_path = "/tmp/stargazer_test_roundtrip.json";

  std::filesystem::copy_file(orig_path, temp_path,
                             std::filesystem::copy_options::overwrite_existing);

  stargazer::configuration cfg1(temp_path);
  auto nodes_before = cfg1.get_nodes("pipeline");
  cfg1.update();  // Serialize back to temp_path

  stargazer::configuration cfg2(temp_path);
  auto nodes_after = cfg2.get_nodes("pipeline");

  std::filesystem::remove(temp_path);

  ASSERT_EQ(nodes_before.size(), nodes_after.size());
  for (size_t i = 0; i < nodes_before.size(); ++i) {
    EXPECT_EQ(nodes_before[i].name, nodes_after[i].name);
    EXPECT_EQ(nodes_before[i].get_type(), nodes_after[i].get_type());
  }
}

}  // namespace
