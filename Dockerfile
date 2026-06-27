FROM nvcr.io/nvidia/isaac-sim:5.1.0
RUN /isaac-sim/kit/python/bin/python3 -m pip install yourdfpy scipy trimesh
WORKDIR /home/kourosh/Paper#3_Simulation/

