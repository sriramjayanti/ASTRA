"""
ASTRA 3D Scene Manager.
Coordinates the active 3D visualization scene corresponding to active pipeline stage or user timeline selection.
"""

from typing import Dict, Optional, Any
import pyqtgraph.opengl as gl

from .base_scene import BaseScene
from .raw_signal_scene import RawSignalScene
from .dsp_scene import DSPScene
from .modulation_scene import ModulationScene
from .baud_scene import BaudScene
from .candidate_scene import CandidateScene
from .sync_scene import SyncScene
from .constellation_scene import ConstellationScene
from .demodulation_scene import DemodulationScene
from .interleaver_scene import InterleaverScene
from .fec_scene import FECScene
from .validation_scene import ValidationScene
from .bitstream_scene import BitstreamScene
from .structure_scene import StructureScene
from .payload_scene import PayloadScene
from .explainability_scene import ExplainabilityScene


class SignalWorldSceneManager:
    """Manages scene instantiation, data routing, and seamless transitions."""

    def __init__(self, view_widget: gl.GLViewWidget):
        self.view = view_widget
        self.scenes: Dict[int, BaseScene] = {}
        self.active_scene: Optional[BaseScene] = None
        self.active_stage_id: int = 1
        self._initialize_scenes()

    def _initialize_scenes(self):
        """Pre-constructs scenes for Stages 1 to 15."""
        scene_classes = {
            1: RawSignalScene,
            2: DSPScene,
            3: ModulationScene,
            4: BaudScene,
            5: CandidateScene,
            6: SyncScene,
            7: ConstellationScene,
            8: InterleaverScene,
            9: FECScene,
            10: ValidationScene,
            11: CandidateScene,       # Stage 11 Scorer uses candidate universe
            12: BitstreamScene,
            13: StructureScene,
            14: PayloadScene,
            15: ExplainabilityScene
        }

        for sid, scls in scene_classes.items():
            scene = scls(self.view)
            scene.initialize()
            self.scenes[sid] = scene

        # Default start on Stage 1
        self.switch_to_stage(1)

    def switch_to_stage(self, stage_id: int):
        """Switches active 3D scene cleanly without GPU memory leaks."""
        if stage_id == self.active_stage_id and self.active_scene is not None and self.active_scene.is_active:
            return

        if self.active_scene:
            self.active_scene.exit()

        self.active_stage_id = stage_id
        target_scene = self.scenes.get(stage_id)
        if target_scene:
            self.active_scene = target_scene
            self.active_scene.enter()

    def feed_stage_data(self, stage_id: int, data: Dict[str, Any]):
        """Passes real pipeline results to designated scene."""
        scene = self.scenes.get(stage_id)
        if scene:
            scene.load_data(data)

    def update(self, dt: float):
        """Tick active scene animation."""
        if self.active_scene and self.active_scene.is_active:
            self.active_scene.update(dt)
