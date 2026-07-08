#!/usr/bin/env bash
# photogrammetry.sh -- fuse the 17 all-LEDs-on endoscope photos into ONE 3D model
# ============================================================================
# Structure-from-Motion + Multi-View Stereo with COLMAP. Uses the 17 overlapping
# reference photos (p{loc}-1.jpg = all LEDs on) to solve true 3D geometry and
# output a single dense mesh you can register to the CT as one object.
#
# Requires COLMAP (CPU build is fine here):  sudo apt-get install -y colmap
# Run:  bash photogrammetry.sh
set -e

ROOT=/sessions/gifted-nifty-davinci/mnt/LuminaBone
WORK=$ROOT/registration/sfm
IMG=$WORK/images
mkdir -p "$IMG"

# 1. collect the all-on reference photos (one per location)
for f in "$ROOT"/bone_picture/p*-1.jpg; do cp "$f" "$IMG/"; done
echo "collected $(ls "$IMG" | wc -l) images"

DB=$WORK/database.db
rm -f "$DB"

# 2. feature extraction (single camera model; endoscope -> radial distortion)
colmap feature_extractor \
    --database_path "$DB" --image_path "$IMG" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.max_image_size 1600 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

# 3. exhaustive matching (only 17 images, so match all pairs)
colmap exhaustive_matcher --database_path "$DB" \
    --SiftMatching.guided_matching 1

# 4. sparse reconstruction (camera poses + sparse points)
mkdir -p "$WORK/sparse"
colmap mapper --database_path "$DB" --image_path "$IMG" \
    --output_path "$WORK/sparse"

# 5. dense: undistort -> patchmatch stereo -> fusion -> mesh
mkdir -p "$WORK/dense"
colmap image_undistorter --image_path "$IMG" \
    --input_path "$WORK/sparse/0" --output_path "$WORK/dense" --output_type COLMAP
colmap patch_match_stereo --workspace_path "$WORK/dense" \
    --PatchMatchStereo.geom_consistency 1
colmap stereo_fusion --workspace_path "$WORK/dense" \
    --output_path "$WORK/dense/fused.ply"
colmap poisson_mesher --input_path "$WORK/dense/fused.ply" \
    --output_path "$WORK/bone_photogrammetry.ply"

echo "DONE -> $WORK/bone_photogrammetry.ply  (single fused model to register to CT)"
