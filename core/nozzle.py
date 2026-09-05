from .constants import GAMMA,CP
def nozzle_model(Tt5,Pt5,ambient_pressure=101325.0,discharge_coefficient=0.98,isentropic_efficiency=0.95):
 pr=ambient_pressure/Pt5; crit=(2/(GAMMA+1))**(GAMMA/(GAMMA-1)); P9=Pt5*crit if pr<=crit else ambient_pressure; T9s=Tt5*(P9/Pt5)**((GAMMA-1)/GAMMA); T9=Tt5-isentropic_efficiency*(Tt5-T9s); V=discharge_coefficient*(2*CP*(Tt5-T9s))**0.5 if Tt5>T9s else 0.0; return {"T9":T9,"V9":V,"P9":P9}
