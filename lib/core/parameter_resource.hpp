#pragma once

#include "coalsack/core/graph_node.h"
#include "parameters.hpp"

namespace stargazer {

class parameter_resource : public coalsack::resource_base {
  std::shared_ptr<parameters_t> parameters_;

 public:
  explicit parameter_resource(std::shared_ptr<parameters_t> parameters)
      : parameters_(std::move(parameters)) {}

  static std::string resource_name() { return "parameter_resource"; }

  std::string get_name() const override { return resource_name(); }

  parameters_t& get_parameters() { return *parameters_; }
  const parameters_t& get_parameters() const { return *parameters_; }
};

}  // namespace stargazer
