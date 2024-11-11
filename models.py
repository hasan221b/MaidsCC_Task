from pydantic import BaseModel

class Device(BaseModel):
    id:int
    battery_power:int
    blue:int	
    clock_speed:int
    dual_sim:int
    fc:int
    four_g:int
    int_memory:int
    m_dep:int
    mobile_wt:int
    n_cores:int
    pc:int
    px_height:int	
    px_width:int	
    ram	:int
    sc_h:int
    sc_w:int
    talk_time:int
    three_g:int
    touch_screen:int	
    wifi:int
