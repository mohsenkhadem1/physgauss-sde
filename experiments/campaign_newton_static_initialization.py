"""Static proximal initial-iterate correction; never changes inertial objective."""
def accepted_start_solver_class(base_solver,wp):
    class AcceptedStartSolver(base_solver):
        def _initialize_particles(self,state_in,state_out,contacts,dt):
            if self.model.body_count or contacts is not None or self.particle_enable_self_contact or getattr(self,'rigid_soft_enable_dat',False):
                raise ValueError('Accepted-start adapter declared for contact-free particles only')
            super()._initialize_particles(state_in,state_out,contacts,dt)
            # Base fills inertia = q_prev + h*v + h²*(g+f/m), unchanged.
            # Base also moves the iterate to that distant predictor. Restore ONLY
            # the iterate; the original BE/proximal force and Hessian stay intact.
            probe=getattr(self,'initialization_probe',None)
            if probe is not None:
                predicted=state_in.particle_q.numpy().copy();accepted=self.particle_q_prev.numpy().copy();target=self.inertia.numpy().copy()
            wp.copy(state_in.particle_q,self.particle_q_prev)
            self.particle_displacements.zero_()
            if probe is not None:
                probe(predicted,accepted,target,self.inertia.numpy(),state_in.particle_q.numpy(),self.pos_prev_collision_detection.numpy(),self.particle_displacements.numpy())
                self.initialization_probe=None
    return AcceptedStartSolver
