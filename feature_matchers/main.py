import argparse
from pathlib import Path

import torch

from LightGlue.lightglue_handler import LightGlueDataHandler
from LightGlue.lightglue_matcher import LightGlueFeatureMatcher
from ORB_matcher.ORB_handler import ORBDataHandler
from ORB_matcher.ORB_matcher import ORBFeatureMatcher
from SIFT_matcher.SIFT_handler import SIFTDataHandler
from SIFT_matcher.SIFT_matcher import SIFTFeatureMatcher
from SuperGlue.superglue_handler import SuperGlueDataHandler
from SuperGlue.superglue_matcher import SuperGlueFeatureMatcher

torch.set_grad_enabled(False)


def match(pairs, data_handler, matcher, use_orb_sift=True):
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        viz_path = "{}_{}_matches.{}".format(stem0, stem1, "png")
        if use_orb_sift:
            img0 = data_handler.read_image(name0)
            img1 = data_handler.read_image(name1)
            features0 = matcher.extract_features(img0)
            features1 = matcher.extract_features(img1)
            matches = matcher.match_features(features0, features1)
            matches = {
                "kpts0": features0["keypoints"],
                "kpts1": features1["keypoints"],
                "matches": matches,
            }
        else:
            img0, inp0 = data_handler.read_image(name0)
            img1, inp1 = data_handler.read_image(name1)
            features0 = matcher.extract_features(inp0)
            features1 = matcher.extract_features(inp1)
            mkpts0, mkpts1 = matcher.match_features(features0, features1)
            matches = {
                "mkpts0": mkpts0,
                "mkpts1": mkpts1,
            }
        matches.update({"image0": img0, "image1": img1, "path": viz_path})
        data_handler.save_macthes(matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image pair matching with SuperGlue / LightGlue / ORB / SIFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir", type=str, help="Path to the directory that contains the images"
    )
    parser.add_argument(
        "--input_pairs", type=str, help="Path to the list of image pairs"
    )
    parser.add_argument(
        "--matcher",
        choices={"orb", "sift", "superglue", "lightglue"},
        type=str,
        help="Matcher to choose from [orb, sift, superglue, lightglue]",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the directory in which the visualization images are written",
    )
    parser.add_argument(
        "--path_to_weights",
        type=str,
        help="SuperGlue weights",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=[640, 480],
        help="Resize the input image before running SuperGlue",
    )

    args = parser.parse_args()
    with open(args.input_pairs, "r") as f:
        pairs = [l.split() for l in f.readlines()]
    if args.matcher == "superglue":
        data_handler = SuperGlueDataHandler(
            args.input_dir, args.output_dir, args.resize
        )
        superglue_matcher = SuperGlueFeatureMatcher(args.path_to_weights)
        match(pairs, data_handler, superglue_matcher, use_orb_sift=False)
    elif args.matcher == "lightglue":
        data_handler = LightGlueDataHandler(args.input_dir, args.output_dir)
        lightglue_matcher = LightGlueFeatureMatcher()
        match(pairs, data_handler, lightglue_matcher, use_orb_sift=False)
    elif args.matcher == "orb":
        data_handler = ORBDataHandler(args.input_dir, args.output_dir)
        orb_matcher = ORBFeatureMatcher()
        match(pairs, data_handler, orb_matcher)
    elif args.matcher == "sift":
        data_handler = SIFTDataHandler(args.input_dir, args.output_dir)
        sift_matcher = SIFTFeatureMatcher()
        match(pairs, data_handler, sift_matcher)
