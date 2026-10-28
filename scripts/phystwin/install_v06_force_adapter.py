"""Make an isolated upstream PhysTwin copy with opt-in prescribed nodal forces."""
from pathlib import Path
import difflib,json,shutil
ROOT=Path(__file__).resolve().parents[2];source=ROOT/'runs/third_party/PhysTwin';dest=ROOT/'runs/third_party/PhysTwin-v06'
def main():
 dest.mkdir(parents=True,exist_ok=True)
 for folder in ['qqtt','configs']:
  if not (dest/folder).exists():shutil.copytree(source/folder,dest/folder)
 shutil.copyfile(source/'LICENSE',dest/'LICENSE')
 rel=Path('qqtt/model/diff_simulator/spring_mass_warp.py');old=(source/rel).read_text();new=old
 assert 'getattr(cfg, "gravity_scale", 1.0)' in old,'Prepare the existing checkout with controlled-physics.patch first; this adapter supplies gravity in its explicit force array.'
 kernel='''@wp.kernel
def add_prescribed_forces(start: wp.array(dtype=wp.vec3), end: wp.array(dtype=wp.vec3), alpha: float, force: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    force[i] = force[i] + (1.0-alpha)*start[i] + alpha*end[i]


'''
 new=new.replace('@wp.kernel\ndef update_vel_from_force(',kernel+'@wp.kernel\ndef update_vel_from_force(',1)
 anchor='        self.num_object_points = num_object_points\n';assert new.count(anchor)==1
 new=new.replace(anchor,anchor+'''        self.external_force_trajectory = None
        if getattr(cfg, "external_force_path", None) is not None:
            import numpy as np
            self.external_force_trajectory = torch.tensor(np.load(cfg.external_force_path)["forces"], dtype=torch.float32, device=self.device)
            assert self.external_force_trajectory.shape[1:] == (num_object_points, 3)
            self.wp_external_start = wp.zeros(num_object_points, dtype=wp.vec3, requires_grad=False)
            self.wp_external_end = wp.zeros(num_object_points, dtype=wp.vec3, requires_grad=False)
''',1)
 anchor='    def set_controller_target(self, frame_idx, pure_inference=False):\n';assert new.count(anchor)==1
 new=new.replace(anchor,anchor+'''        if self.external_force_trajectory is not None:
            wp.launch(copy_vec3, dim=self.num_object_points, inputs=[self.external_force_trajectory[frame_idx-1]], outputs=[self.wp_external_start])
            wp.launch(copy_vec3, dim=self.num_object_points, inputs=[self.external_force_trajectory[frame_idx]], outputs=[self.wp_external_end])
''',1)
 anchor='            # Calculate the spring forces\n';assert new.count(anchor)==1
 new=new.replace(anchor,'''            if self.external_force_trajectory is not None:
                wp.launch(add_prescribed_forces, dim=self.num_object_points, inputs=[self.wp_external_start, self.wp_external_end, float(i+1)/float(self.num_substeps)], outputs=[self.wp_states[i].wp_vertice_forces])

'''+anchor,1)
 (dest/rel).write_text(new)
 (ROOT/'scripts/phystwin/v06-prescribed-forces.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+str(rel),tofile='b/'+str(rel),n=0)))
 print(dest)
if __name__=='__main__':main()
