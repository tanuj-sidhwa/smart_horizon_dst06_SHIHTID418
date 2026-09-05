from .constants import GAMMA,R
def inlet_model(ambient_pressure=101325.0,ambient_temperature=288.15,ambient_velocity=0.0,inlet_pressure_recovery=0.98):
 a=(GAMMA*R*ambient_temperature)**0.5; M=ambient_velocity/a
 return {"Tt2":ambient_temperature*(1+(GAMMA-1)/2*M*M),"Pt2":ambient_pressure*(1+(GAMMA-1)/2*M*M)**(GAMMA/(GAMMA-1))*inlet_pressure_recovery}
