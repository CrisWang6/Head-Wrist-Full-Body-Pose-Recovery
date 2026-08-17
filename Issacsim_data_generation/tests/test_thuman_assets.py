from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from geosim.appearance import prepare_one_appearance_asset
from geosim.thuman_assets import (
    build_scan_subject_index,
    build_smplx_subject_index,
    select_subjects,
)


class ThumanAssetPreparationTest(unittest.TestCase):
    def test_archive_subject_indexes(self) -> None:
        smplx_members = [
            "THuman2.1_Smpl-X/0007/mesh_smplx.obj",
            "THuman2.1_Smpl-X/0007/smplx_param.pkl",
            "THuman2.1_Smpl-X/0012/mesh_smplx.obj",
            "THuman2.1_Smpl-X/0012/smplx_param.pkl",
        ]
        scan_members = [
            "THuman2.1_Release/0007/0007.obj",
            "THuman2.1_Release/0007/material0.jpeg",
            "THuman2.1_Release/0012/0012.obj",
        ]
        smplx_index = build_smplx_subject_index(smplx_members)
        scan_index = build_scan_subject_index(scan_members)
        self.assertEqual(sorted(smplx_index), ["0007", "0012"])
        self.assertEqual(scan_index["0007"].root, "THuman2.1_Release/0007")
        self.assertEqual(select_subjects(smplx_index, count=1), ["0007"])

    def test_prepare_one_asset_pairs_smplx_and_texture_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smplx_dir = root / "smplx" / "0007"
            scan_dir = root / "scans" / "0007"
            out_dir = root / "appearances"
            smplx_dir.mkdir(parents=True)
            scan_dir.mkdir(parents=True)
            (smplx_dir / "mesh_smplx.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            (smplx_dir / "smplx_param.pkl").write_bytes(b"fake")
            (scan_dir / "0007.obj").write_text(
                "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nusemtl m\nf 1 2 3\n",
                encoding="utf-8",
            )
            (scan_dir / "material.mtl").write_text("newmtl m\nmap_Kd material0.jpeg\n", encoding="utf-8")
            (scan_dir / "material0.jpeg").write_bytes(b"fake")

            asset = prepare_one_appearance_asset(
                source_dir=smplx_dir,
                output_root=out_dir,
                thuman_scans_root=scan_dir.parent,
                overwrite=False,
                strict_textures=True,
            )
            self.assertEqual(asset.subject, "0007")
            self.assertTrue(asset.smplx_mesh.exists())
            self.assertTrue(asset.smplx_params.exists())
            self.assertIsNotNone(asset.textured_mesh)
            self.assertTrue((asset.asset_dir / "icon_textured" / "material0.jpeg").exists())


if __name__ == "__main__":
    unittest.main()
