from .constants import GAMMA,CP,MECHANICAL_EFFICIENCY
def turbine_model(Tt4,Pt4,compressor_specific_work,turbine_efficiency):
 work=compressor_specific_work/MECHANICAL_EFFICIENCY; Tt5=Tt4-work/CP; Tt5s=Tt4-(work/CP)/turbine_efficiency; pr=(Tt4/Tt5s)**(GAMMA/(GAMMA-1)); return {"Tt5":Tt5,"Pt5":Pt4/pr}
