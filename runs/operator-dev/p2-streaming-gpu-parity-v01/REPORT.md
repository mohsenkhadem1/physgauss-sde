# Streaming P2 GPU parity

The single authorized GPU2 Q3 12,000-tetrahedron check passed. The streaming solve from rest agrees with the immutable monolithic solution to **3.897e-15 m** across every node. Original free-force residual is **9.283e-13 N**, global force defect **1.210e-12 N**, moment defect **1.748e-14 N m**, and clamp error zero. Minimum J is 0.9174; the sufficient global positive-J lower bound is 0.4324.

At the frozen field, streamed and monolithic energy, gradient, full determinant-cross HVP, node blocks and full consistent mass satisfy the predeclared operator gates. Worker time was 19.41 s; peak CUDA allocation was 409,053,184 bytes, including simultaneous operator-control objects. This is implementation parity at one fixed discretization, not spatial, temporal or inference qualification.

The controller and worker terminated and were reaped with exit zero. GPU2 was released. No Q80 fine study was launched.
