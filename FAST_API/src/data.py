from pydantic import BaseModel
from typing import List, Dict


class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class WineResponse(BaseModel):
    response: int 

class WineLabelResponse(WineResponse):
    pass  

class WineProbaResponse(BaseModel):
    probabilities: List[float]
    class_names: List[str]
    predicted_label: int

class MinimalInput(BaseModel):
    features: Dict[str, float]  

class MinimalResponse(BaseModel):
    predicted_label: int
    probabilities: List[float]
    used_features: List[str]
    class_names: List[str]
