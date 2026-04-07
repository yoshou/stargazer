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
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 2u);

  auto cam0_source = std::find_if(nodes.begin(), nodes.end(),
                                  [](const auto& n) { return n.name == "cam0/source"; });
  auto cam0_decoder = std::find_if(nodes.begin(), nodes.end(),
                                   [](const auto& n) { return n.name == "cam0/decoder"; });

  ASSERT_NE(cam0_source, nodes.end()) << "Expected node 'cam0/source' not found";
  ASSERT_NE(cam0_decoder, nodes.end()) << "Expected node 'cam0/decoder' not found";

  ASSERT_EQ(cam0_source->subgraph_instance, "cam0");
  ASSERT_EQ(cam0_decoder->subgraph_instance, "cam0");

  // Local input reference "source" must be rewritten to "cam0/source"
  ASSERT_TRUE(cam0_decoder->inputs.count("default") > 0);
  EXPECT_EQ(cam0_decoder->inputs.at("default"), "cam0/source");
}

// ---------------------------------------------------------------------------
// TC2: Direct nodes (no extends)
// Pipeline subgraph has inline nodes. Names must not be prefixed.
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC2_DirectNodes) {
  stargazer::configuration cfg(fixture("test_direct_nodes.json"));
  auto nodes = cfg.get_nodes();

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
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 1u);
  EXPECT_EQ(nodes[0].name, "cam0/loader");

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
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 1u);
  EXPECT_EQ(nodes[0].name, "cam0/loader");

  // Override should take effect over template value
  EXPECT_EQ(nodes[0].get_param<std::string>("db_path"), "override_db");
}

// ---------------------------------------------------------------------------
// TC5: Cross-subgraph input reference
// Instance override sets inputs like "ext_sg/source_node" (slash notation).
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC5_CrossRefSlash) {
  stargazer::configuration cfg(fixture("test_cross_ref.json"));
  auto nodes = cfg.get_nodes();

  // Find sg1/processor (prefixed because it extends consumer_template)
  auto proc = std::find_if(nodes.begin(), nodes.end(),
                           [](const auto& n) { return n.name == "sg1/processor"; });
  ASSERT_NE(proc, nodes.end()) << "Expected node 'sg1/processor' not found";

  ASSERT_TRUE(proc->inputs.count("cam0") > 0);
  ASSERT_TRUE(proc->inputs.count("cam1") > 0);

  // '.' in "ext_sg.source_node" must become '/'
  EXPECT_EQ(proc->inputs.at("cam0"), "ext_sg/source_node");
  EXPECT_EQ(proc->inputs.at("cam1"), "ext_sg/other_node");
}

// ---------------------------------------------------------------------------
// TC6: Multiple subgraphs (extends + direct)
// cam0 extends camera_template (2 nodes, prefixed).
// reconstructor has a direct node (no prefix).
// Total: 3 nodes.
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC6_MultipleSubgraphs) {
  stargazer::configuration cfg(fixture("test_multiple_subgraphs.json"));
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 3u);

  auto cam0_loader = std::find_if(nodes.begin(), nodes.end(),
                                  [](const auto& n) { return n.name == "cam0/loader"; });
  auto cam0_decoder = std::find_if(nodes.begin(), nodes.end(),
                                   [](const auto& n) { return n.name == "cam0/decoder"; });
  auto reconstruct = std::find_if(nodes.begin(), nodes.end(),
                                  [](const auto& n) { return n.name == "reconstruct_node"; });

  ASSERT_NE(cam0_loader, nodes.end()) << "Expected 'cam0/loader' not found";
  ASSERT_NE(cam0_decoder, nodes.end()) << "Expected 'cam0/decoder' not found";
  ASSERT_NE(reconstruct, nodes.end()) << "Expected 'reconstruct_node' not found";

  // cam0/decoder's local input must be prefixed
  EXPECT_EQ(cam0_decoder->inputs.at("default"), "cam0/loader");

  // reconstructor's input must be preserved as-is (cross-subgraph, no dot)
  EXPECT_EQ(reconstruct->inputs.at("cam0"), "cam0/decoder");
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
  auto nodes_before = cfg1.get_nodes();
  cfg1.update();  // Serialize back to temp_path

  stargazer::configuration cfg2(temp_path);
  auto nodes_after = cfg2.get_nodes();

  std::filesystem::remove(temp_path);

  ASSERT_EQ(nodes_before.size(), nodes_after.size());
  for (size_t i = 0; i < nodes_before.size(); ++i) {
    EXPECT_EQ(nodes_before[i].name, nodes_after[i].name);
    EXPECT_EQ(nodes_before[i].get_type(), nodes_after[i].get_type());
  }
}

// ---------------------------------------------------------------------------
// TC8: Nested subgraph expansion via group template
// group_template has subgraphs: [src→src_template, sink→sink_template]
// Instance "cam0" extends "group_template".
// Case 1a: template's internal name space is self-contained; outer prefix
// ("cam0") is NOT propagated into nested subgraphs.
// Expected: 2 nodes named "src_loader" and "sink_decoder".
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC8_NestedExpansion) {
  stargazer::configuration cfg(fixture("test_nested_expansion.json"));
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 2u);

  auto src_loader = std::find_if(nodes.begin(), nodes.end(),
                                 [](const auto& n) { return n.name == "src/loader"; });
  auto sink_decoder = std::find_if(nodes.begin(), nodes.end(),
                                   [](const auto& n) { return n.name == "sink/decoder"; });

  ASSERT_NE(src_loader, nodes.end()) << "Expected 'src/loader' not found";
  ASSERT_NE(sink_decoder, nodes.end()) << "Expected 'sink/decoder' not found";
}

// ---------------------------------------------------------------------------
// TC9: Nested subgraph param propagation
// Instance "cam0" extends "group" (which has nested subgraphs) with db_path="cam0_db".
// Case 1a: outer prefix not propagated. Node name is "src_loader".
// Expected: inner node "src_loader" gets db_path="cam0_db".
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC9_NestedParamPropagation) {
  stargazer::configuration cfg(fixture("test_nested_params.json"));
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 1u);
  EXPECT_EQ(nodes[0].name, "src/loader");
  EXPECT_EQ(nodes[0].get_param<std::string>("db_path"), "cam0_db");
}

// ---------------------------------------------------------------------------
// TC10: Inline nested subgraphs in pipeline instance (no extends on outer)
// Pipeline instance "cam0" has inline subgraphs: [part1, part2] each extending node_template.
// Expected: 2 nodes "cam0_part1_loader" and "cam0_part2_loader".
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC10_InlineNestedSubgraphs) {
  stargazer::configuration cfg(fixture("test_inline_nested.json"));
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 2u);

  auto part1 = std::find_if(nodes.begin(), nodes.end(),
                            [](const auto& n) { return n.name == "cam0/part1/loader"; });
  auto part2 = std::find_if(nodes.begin(), nodes.end(),
                            [](const auto& n) { return n.name == "cam0/part2/loader"; });

  ASSERT_NE(part1, nodes.end()) << "Expected 'cam0/part1/loader' not found";
  ASSERT_NE(part2, nodes.end()) << "Expected 'cam0/part2/loader' not found";
}

// ---------------------------------------------------------------------------
// TC11: Nested roundtrip — load → update() → reload → same nodes
// Uses a group template with two nested subgraphs (part_a, part_b), each
// extending leaf_template which has internal input references.
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC11_NestedRoundtrip) {
  const std::string orig_path = fixture("test_nested_roundtrip.json");
  const std::string temp_path = "/tmp/stargazer_test_nested_roundtrip.json";

  std::filesystem::copy_file(orig_path, temp_path,
                             std::filesystem::copy_options::overwrite_existing);

  stargazer::configuration cfg1(temp_path);
  auto nodes_before = cfg1.get_nodes();
  cfg1.update();

  stargazer::configuration cfg2(temp_path);
  auto nodes_after = cfg2.get_nodes();

  std::filesystem::remove(temp_path);

  ASSERT_EQ(nodes_before.size(), nodes_after.size());
  for (size_t i = 0; i < nodes_before.size(); ++i) {
    EXPECT_EQ(nodes_before[i].name, nodes_after[i].name);
    EXPECT_EQ(nodes_before[i].get_type(), nodes_after[i].get_type());
  }
}

// ---------------------------------------------------------------------------
// TC12: Direct nodes with cross-subgraph slash references
// Subgraph "sync" has direct nodes (no extends). Inputs use slash notation
// like "camera1_sg/decoder".
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC12_DirectNodesDotConversion) {
  stargazer::configuration cfg(fixture("test_direct_dot_ref.json"));
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 2u);

  auto sync_node = std::find_if(nodes.begin(), nodes.end(),
                                [](const auto& n) { return n.name == "approximate_time_sync"; });
  auto cb_node =
      std::find_if(nodes.begin(), nodes.end(), [](const auto& n) { return n.name == "callback"; });

  ASSERT_NE(sync_node, nodes.end()) << "Expected 'approximate_time_sync' not found";
  ASSERT_NE(cb_node, nodes.end()) << "Expected 'callback' not found";

  // Dot in cross-subgraph reference must be converted to slash
  ASSERT_TRUE(sync_node->inputs.count("camera1") > 0);
  ASSERT_TRUE(sync_node->inputs.count("camera2") > 0);
  EXPECT_EQ(sync_node->inputs.at("camera1"), "camera1_sg/decoder");
  EXPECT_EQ(sync_node->inputs.at("camera2"), "camera2_sg/decoder");

  // Local reference (no dot) must remain unchanged
  EXPECT_EQ(cb_node->inputs.at("default"), "approximate_time_sync");
}

// ---------------------------------------------------------------------------
// TC14: Template nested subgraph with direct nodes (no extends)
// Template "combo_template" has nested subgraphs:
//   - part1: extends leaf_template (Case 1b → "part1_source")
//   - aggregator: direct nodes with inputs using "part1.source"
//     (Case 3 → dot→underscore → node names unchanged, input "part1_source")
// Expected: 3 nodes: part1_source, sync_node (no prefix), callback_node (no prefix)
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC14_TemplateNestedDirectNodes) {
  stargazer::configuration cfg(fixture("test_template_nested_direct.json"));
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 3u);

  auto part1_source = std::find_if(nodes.begin(), nodes.end(),
                                   [](const auto& n) { return n.name == "part1/source"; });
  auto sync_node = std::find_if(nodes.begin(), nodes.end(),
                                [](const auto& n) { return n.name == "sync_node"; });
  auto callback_node = std::find_if(nodes.begin(), nodes.end(),
                                    [](const auto& n) { return n.name == "callback_node"; });

  ASSERT_NE(part1_source, nodes.end()) << "Expected 'part1/source' not found";
  ASSERT_NE(sync_node, nodes.end()) << "Expected 'sync_node' not found";
  ASSERT_NE(callback_node, nodes.end()) << "Expected 'callback_node' not found";

  // aggregator is Case 3: inputs' dot→slash applied
  ASSERT_TRUE(sync_node->inputs.count("part1") > 0);
  EXPECT_EQ(sync_node->inputs.at("part1"), "part1/source");

  ASSERT_TRUE(callback_node->inputs.count("default") > 0);
  EXPECT_EQ(callback_node->inputs.at("default"), "sync_node");
}

// ---------------------------------------------------------------------------
// TC15: Template nested subgraph with extends + node param override
// Template "group_template" has nested subgraphs cam1 and cam2, each
// extending "loader_template" (blob + decode) with per-camera topic_name.
// Expected: 4 nodes cam1_blob (topic=image_cam1), cam1_decode,
//           cam2_blob (topic=image_cam2), cam2_decode.
// ---------------------------------------------------------------------------
TEST(ConfigGetNodes, TC15_TemplateNestedNodeOverride) {
  stargazer::configuration cfg(fixture("test_template_nested_override.json"));
  auto nodes = cfg.get_nodes();

  ASSERT_EQ(nodes.size(), 4u);

  auto cam1_blob = std::find_if(nodes.begin(), nodes.end(),
                                [](const auto& n) { return n.name == "cam1/blob"; });
  auto cam1_decode = std::find_if(nodes.begin(), nodes.end(),
                                  [](const auto& n) { return n.name == "cam1/decode"; });
  auto cam2_blob = std::find_if(nodes.begin(), nodes.end(),
                                [](const auto& n) { return n.name == "cam2/blob"; });
  auto cam2_decode = std::find_if(nodes.begin(), nodes.end(),
                                  [](const auto& n) { return n.name == "cam2/decode"; });

  ASSERT_NE(cam1_blob, nodes.end()) << "Expected 'cam1/blob' not found";
  ASSERT_NE(cam1_decode, nodes.end()) << "Expected 'cam1/decode' not found";
  ASSERT_NE(cam2_blob, nodes.end()) << "Expected 'cam2/blob' not found";
  ASSERT_NE(cam2_decode, nodes.end()) << "Expected 'cam2/decode' not found";

  // Node override must apply per-camera topic_name
  EXPECT_EQ(cam1_blob->get_param<std::string>("topic_name"), "image_cam1");
  EXPECT_EQ(cam2_blob->get_param<std::string>("topic_name"), "image_cam2");

  // Local input reference inside template must be prefixed
  EXPECT_EQ(cam1_decode->inputs.at("default"), "cam1/blob");
  EXPECT_EQ(cam2_decode->inputs.at("default"), "cam2/blob");
}

}  // namespace
