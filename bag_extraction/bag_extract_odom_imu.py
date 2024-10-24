import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader

ODOM_TOPIC = "/aft_mapped_to_init_CORRECTED"
TF_TOPIC = "/tf"
IMU_TOPIC = "/vn100/imu"


def extract_odometry(reader, topic, output_file):
    connections = [x for x in reader.connections if x.topic == topic]
    with open(output_file, "w") as f:
        f.write(
            "%time,"
            "field.header.seq,"
            "field.header.stamp,"
            "field.header.frame_id,"
            "field.child_frame_id,"
            "field.pose.pose.position.x,"
            "field.pose.pose.position.y,"
            "field.pose.pose.position.z,"
            "field.pose.pose.orientation.x,"
            "field.pose.pose.orientation.y,"
            "field.pose.pose.orientation.z,"
            "field.pose.pose.orientation.w,"
            f'{",".join([f"field.pose.covariance{i}" for i in range(36)])},'
            "field.twist.twist.linear.x,"
            "field.twist.twist.linear.y,"
            "field.twist.twist.linear.z,"
            "field.twist.twist.angular.x,"
            "field.twist.twist.angular.y,"
            "field.twist.twist.angular.z,"
            f'{",".join([f"field.twist.covariance{i}" for i in range(36)])}'
            "\n"
        )
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            time = str(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)
            f.write(
                f"{time},"
                f"{msg.header.seq},"
                f"{msg.header.stamp.nanosec + msg.header.stamp.sec * 10 ** 9},"
                f"{msg.header.frame_id},"
                f"{msg.child_frame_id},"
                f"{msg.pose.pose.position.x},"
                f"{msg.pose.pose.position.y},"
                f"{msg.pose.pose.position.z},"
                f"{msg.pose.pose.orientation.x},"
                f"{msg.pose.pose.orientation.y},"
                f"{msg.pose.pose.orientation.z},"
                f"{msg.pose.pose.orientation.w},"
                f'{",".join([str(i) for i in msg.pose.covariance])},'  # 36
                f"{msg.twist.twist.linear.x},"
                f"{msg.twist.twist.linear.y},"
                f"{msg.twist.twist.linear.z},"
                f"{msg.twist.twist.angular.x},"
                f"{msg.twist.twist.angular.y},"
                f"{msg.twist.twist.angular.z},"
                f'{",".join([str(i) for i in msg.twist.covariance])}'  # 36
                f"\n"
            )


def extract_imu(reader, topic, output_path):
    connections = [x for x in reader.connections if x.topic == topic]
    with open(output_path / "linear_acceleration.txt", "w") as f:
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            time = str(msg.header.stamp.sec) + str(msg.header.stamp.nanosec).rjust(
                9, "0"
            )
            f.write(
                f"{time} {msg.linear_acceleration.x} {msg.linear_acceleration.y} {msg.linear_acceleration.z}\n"
            )
    with open(output_path / "angular_velocity.txt", "w") as f:
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            time = str(msg.header.stamp.sec) + str(msg.header.stamp.nanosec).rjust(
                9, "0"
            )
            f.write(
                f"{time} {msg.angular_velocity.x} {msg.angular_velocity.y} {msg.angular_velocity.z}\n"
            )


def extract_tf(reader, topic, output_file):
    connections = [x for x in reader.connections if x.topic == topic]
    with open(output_file, "w") as f:
        f.write(
            "%time,"
            "field.header.seq,"
            "field.header.stamp,"
            "field.header.frame_id,"
            "field.child_frame_id,"
            "field.transform.translation.x,"
            "field.transform.translation.y,"
            "field.transform.translation.z,"
            "field.transform.rotation.x,"
            "field.transform.rotation.y,"
            "field.transform.rotation.z,"
            "field.transform.rotation.w"
            "\n"
        )
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if len(msg.transforms) != 0:
                time = str(
                    msg.transforms[0].header.stamp.sec * 10**9
                    + msg.transforms[0].header.stamp.nanosec
                )
                f.write(
                    f"{time},"
                    f"{msg.transforms[0].header.seq},"
                    f"{msg.transforms[0].header.stamp.nanosec + msg.transforms[0].header.stamp.sec * 10 ** 9},"
                    f"{msg.transforms[0].header.frame_id},"
                    f"{msg.transforms[0].child_frame_id},"
                    f"{msg.transforms[0].transform.translation.x},"
                    f"{msg.transforms[0].transform.translation.y},"
                    f"{msg.transforms[0].transform.translation.z},"
                    f"{msg.transforms[0].transform.rotation.x},"
                    f"{msg.transforms[0].transform.rotation.y},"
                    f"{msg.transforms[0].transform.rotation.z},"
                    f"{msg.transforms[0].transform.rotation.w}"
                    f"\n"
                )


def process_bag(bag_path, output_path):
    with AnyReader([bag_path]) as reader:
        print(f"Extracting odometry from {bag_path}")
        extract_odometry(
            reader,
            ODOM_TOPIC,
            output_path / f"odom{ODOM_TOPIC.replace('/', '_')}.csv",
        )
        print(f"Extracting tf from {bag_path}")
        extract_tf(reader, TF_TOPIC, output_path / "tf.csv")

        print(f"Extracting IMU data from {bag_path}")
        extract_imu(reader, IMU_TOPIC, output_path)

        print(f"Finished processing {bag_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        help="Path to the input ROSBag file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Path to the output directory",
    )
    args = parser.parse_args()
    bag_path = args.input_file
    output_path = args.output_dir
    output_path.mkdir(exist_ok=True)
    process_bag(bag_path, output_path)
