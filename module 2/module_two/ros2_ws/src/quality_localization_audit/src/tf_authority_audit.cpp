#include <chrono>
#include <filesystem>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace
{
std::string gid_hex(const uint8_t * data)
{
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = 0; i < RMW_GID_STORAGE_SIZE; ++i) {
    out << std::setw(2) << static_cast<unsigned>(data[i]);
  }
  return out.str();
}
std::string json_string(const std::string & value)
{
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {out << '\\' << c;}
    else if (c < 0x20) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c);
    } else {out << c;}
  }
  out << '"';
  return out.str();
}
std::string normalize_frame(const std::string & frame)
{
  const auto first = frame.find_first_not_of('/');
  return first == std::string::npos ? "" : frame.substr(first);
}
}  // namespace

class TfAuthorityAudit : public rclcpp::Node
{
public:
  TfAuthorityAudit() : Node("tf_authority_audit")
  {
    executable_ = std::filesystem::read_symlink("/proc/self/exe").string();
    output_ = create_publisher<std_msgs::msg::String>(
      "/benchmark/tf_authorities", rclcpp::QoS(1).reliable().transient_local());
    auto callback = [this](const tf2_msgs::msg::TFMessage::SharedPtr msg,
        const rclcpp::MessageInfo & info) {observe(msg, info);};
    dynamic_ = create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", rclcpp::QoS(100).best_effort(), callback);
    fixed_ = create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf_static", rclcpp::QoS(100).reliable().transient_local(), callback);
    timer_ = create_wall_timer(std::chrono::milliseconds(500), [this]() {publish();});
  }

private:
  void collect_publishers()
  {
    for (const auto * topic : {"/tf", "/tf_static"}) {
      for (const auto & info : get_publishers_info_by_topic(topic)) {
        // Keep identity after exit so short-lived duplicate owners stay visible.
        publishers_[gid_hex(info.endpoint_gid().data())] =
          std::make_pair(info.node_name(), info.node_namespace());
      }
    }
  }
  void observe(const tf2_msgs::msg::TFMessage::SharedPtr & msg,
    const rclcpp::MessageInfo & info)
  {
    const auto gid = gid_hex(info.get_rmw_message_info().publisher_gid.data);
    if (publishers_.count(gid) == 0) {collect_publishers();}
    for (const auto & transform : msg->transforms) {
      const auto pair = normalize_frame(transform.header.frame_id) + "->" +
        normalize_frame(transform.child_frame_id);
      authorities_[pair].insert(gid);
    }
    ++observed_messages_;
  }
  void publish()
  {
    collect_publishers();
    std::ostringstream json;
    json << "{\"stamp\":" << std::setprecision(17) << now().seconds()
         << ",\"observed_messages\":" << observed_messages_
         << ",\"tf_authorities\":{";
    bool first_pair = true;
    for (const auto & item : authorities_) {
      if (!first_pair) {json << ',';}
      first_pair = false;
      json << json_string(item.first) << ":[";
      bool first_gid = true;
      for (const auto & gid : item.second) {
        if (!first_gid) {json << ',';}
        first_gid = false;
        json << json_string(gid);
      }
      json << ']';
    }
    json << "},\"publishers\":[";
    bool first_publisher = true;
    for (const auto & item : publishers_) {
      if (!first_publisher) {json << ',';}
      first_publisher = false;
      json << "{\"node\":" << json_string(item.second.first)
           << ",\"namespace\":" << json_string(item.second.second)
           << ",\"gid\":" << json_string(item.first) << '}';
    }
    json << "],\"runtime_executable_path\":" << json_string(executable_) << '}';
    std_msgs::msg::String message;
    message.data = json.str();
    output_->publish(message);
  }
  std::string executable_;
  std::map<std::string, std::set<std::string>> authorities_;
  std::map<std::string, std::pair<std::string, std::string>> publishers_;
  uint64_t observed_messages_{0};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr output_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_, fixed_;
  rclcpp::TimerBase::SharedPtr timer_;
};
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TfAuthorityAudit>());
  rclcpp::shutdown();
  return 0;
}
