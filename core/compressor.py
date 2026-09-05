from .constants import GAMMA,CP
def compressor_model(Tt2,Pt2,pressure_ratio,compressor_efficiency):
 Tt3s=Tt2*pressure_ratio**((GAMMA-1)/GAMMA); Tt3=Tt2+(Tt3s-Tt2)/compressor_efficiency
 return {"Tt3":Tt3,"Pt3":Pt2*pressure_ratio,"specific_work":CP*(Tt3-Tt2)}
