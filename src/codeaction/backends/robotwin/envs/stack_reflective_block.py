"""Paired reflective-RGB variant of the upstream ``stack_blocks_two`` task.

The scene geometry, physics, randomization, reference, and success predicate are inherited
unchanged.  Only the upper block's render material changes.  This isolates RGB appearance from
task semantics and must not be described as a realistic reflective-depth failure: RoboTwin's
native depth is the ideal SAPIEN Position buffer and is material-independent.
"""

import sapien

from envs.stack_blocks_two import stack_blocks_two


class stack_reflective_block(stack_blocks_two):

    def load_actors(self):
        super().load_actors()
        for component in self.block2.actor.get_components():
            if not isinstance(component, sapien.render.RenderBodyComponent):
                continue
            for shape in component.render_shapes:
                material = shape.material
                material.base_color = [0.68, 0.72, 0.78, 1.0]
                material.metallic = 1.0
                material.roughness = 0.01
                material.specular = 1.0
