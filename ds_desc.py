titanic = """
## Titanic Dataset Description

[https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)

- pclass: A proxy for socio-economic status (SES)
    - 1st = Upper
    - 2nd = Middle
    - 3rd = Lower
- age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
- sibsp: The dataset defines family relations in this way...
    - Sibling = brother, sister, stepbrother, stepsister
    - Spouse = husband, wife (mistresses and fiancés were ignored)
- parch: The dataset defines family relations in this way...
    - Parent = mother, father
    - Child = daughter, son, stepdaughter, stepson
    - Some children travelled only with a nanny, therefore parch=0 for them.
- title
"""

adult = """
## Adult Dataset Description

[https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)

- target: >50K, <=50K.

- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

"""

california = """
## California Housing Dataset Description

[https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

- MedHouseVal(Target): Median house value for households within a block (measured in US Dollars)
- Longitude: A measure of how far west a house is; a more negative value is farther west, (-124.3, -114.3)
- Latitude: A measure of how far north a house is; a higher value is farther north,  (32.5, 42.5)
- HouseAge: Median age of a house within a block; a lower number is a newer building
- AveRooms Average number of rooms per household
- AveBedrms Average number of bedrooms per household
- Population: Total number of people residing within a block
- AveOccup: Average number of households, a group of people residing within a home unit, for a block
- MedInc: Median income for households within a block of houses (measured in tens of thousands of US Dollars)

- Latitude: [(32.539, 33.87] < (33.87, 34.1] < (34.1, 36.66] < (36.66, 37.81] < (37.81, 41.95]
- Longtitude: [(-124.351, -121.98] < (-121.98, -119.91] < (-119.91, -118.3] < (-118.3, -117.89] < (-117.89, -114.31]]
"""

boston = """
## Boston Housing Dataset Description

[http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html](http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)


- CRIM: per capita crime rate by town
- ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: proportion of non-retail business acres per town.
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX: nitric oxides concentration (parts per 10 million)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centres
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: percen of lower status of the population
- MEDV: Median value of owner-occupied homes in $1000's

"""

ames = """
## AMES Housing Dataset Description

[http://jse.amstat.org/v19n3/decock.pdf](http://jse.amstat.org/v19n3/decock.pdf)

- SalePrice(1,000 K) - the property's sale price in dollars. This is the target variable that you're trying to predict.
- MSSubClass: The building class
- MSZoning: The general zoning classification
- LotFrontage: Linear feet of street connected to property
- LotArea: Lot size in square feet
- Street: Type of road access
- Alley: Type of alley access
- LotShape: General shape of property
- LandContour: Flatness of the property
- Utilities: Type of utilities available
- LotConfig: Lot configuration
- LandSlope: Slope of property
- Neighborhood: Physical locations within Ames city limits
- Condition1: Proximity to main road or railroad
- Condition2: Proximity to main road or railroad (if a second is present)
- BldgType: Type of dwelling
- HouseStyle: Style of dwelling
- OverallQual: Overall material and finish quality
- OverallCond: Overall condition rating
- YearBuilt: Original construction date
- YearRemodAdd: Remodel date
- RoofStyle: Type of roof
- RoofMatl: Roof material
- Exterior1st: Exterior covering on house
- Exterior2nd: Exterior covering on house (if more than one material)
- MasVnrType: Masonry veneer type
- MasVnrArea: Masonry veneer area in square feet
- ExterQual: Exterior material quality
- ExterCond: Present condition of the material on the exterior
- Foundation: Type of foundation
- BsmtQual: Height of the basement
- BsmtCond: General condition of the basement
- BsmtExposure: Walkout or garden level basement walls
- BsmtFinType1: Quality of basement finished area
- BsmtFinSF1: Type 1 finished square feet
- BsmtFinType2: Quality of second finished area (if present)
- BsmtFinSF2: Type 2 finished square feet
- BsmtUnfSF: Unfinished square feet of basement area
- TotalBsmtSF: Total square feet of basement area
- Heating: Type of heating
- HeatingQC: Heating quality and condition
- CentralAir: Central air conditioning
- Electrical: Electrical system
- 1stFlrSF: First Floor square feet
- 2ndFlrSF: Second floor square feet
- LowQualFinSF: Low quality finished square feet (all floors)
- GrLivArea: Above grade (ground) living area square feet
- BsmtFullBath: Basement full bathrooms
- BsmtHalfBath: Basement half bathrooms
- FullBath: Full bathrooms above grade
- HalfBath: Half baths above grade
- Bedroom: Number of bedrooms above basement level
- Kitchen: Number of kitchens
- KitchenQual: Kitchen quality
- TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
- Functional: Home functionality rating
- Fireplaces: Number of fireplaces
- FireplaceQu: Fireplace quality
- GarageType: Garage location
- GarageYrBlt: Year garage was built
- GarageFinish: Interior finish of the garage
- GarageCars: Size of garage in car capacity
- GarageArea: Size of garage in square feet
- GarageQual: Garage quality
- GarageCond: Garage condition
- PavedDrive: Paved driveway
- WoodDeckSF: Wood deck area in square feet
- OpenPorchSF: Open porch area in square feet
- EnclosedPorch: Enclosed porch area in square feet
- 3SsnPorch: Three season porch area in square feet
- ScreenPorch: Screen porch area in square feet
- PoolArea: Pool area in square feet
- PoolQC: Pool quality
- Fence: Fence quality
- MiscFeature: Miscellaneous feature not covered in other categories
- MiscVal: $Value of miscellaneous feature
- MoSold: Month Sold
- YrSold: Year Sold
- SaleType: Type of sale
- SaleCondition: Condition of sale

- MSZoning ['RL', 'RM', 'C (all)', 'FV', 'RH']
- Street ['Pave', 'Grvl']
- Alley ['NA', 'Grvl', 'Pave']
- LotShape ['Reg', 'IR1', 'IR2', 'IR3']
- LandContour ['Lvl', 'Bnk', 'Low', 'HLS']
- Utilities ['AllPub', 'NoSeWa']
- LotConfig ['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3']
- LandSlope ['Gtl', 'Mod', 'Sev']
- Neighborhood ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste']
- Condition1 ['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe']
- Condition2 ['Norm', 'Artery', 'RRNn', 'Feedr', 'PosN', 'PosA', 'RRAn', 'RRAe']
- BldgType ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs']
- HouseStyle ['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin']
- RoofStyle ['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed']
- RoofMatl ['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv', 'Roll', 'ClyTile']
- Exterior1st ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock']
- Exterior2nd ['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng', 'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc', 'AsphShn', 'Stone', 'Other', 'CBlock']
- MasVnrType ['BrkFace', 'None', 'Stone', 'BrkCmn', 0]
- ExterQual ['Gd', 'TA', 'Ex', 'Fa']
- ExterCond ['TA', 'Gd', 'Fa', 'Po', 'Ex']
- Foundation ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone']
- BsmtQual ['NA', 'Gd', 'TA', 'Ex', 'Fa']
- BsmtCond ['NA', 'TA', 'Gd', 'Fa', 'Po']
- BsmtExposure ['NA', 'No', 'Gd', 'Mn', 'Av']
- BsmtFinType1 ['NA', 'GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ']
- BsmtFinType2 ['NA', 'Unf', 'BLQ', 'ALQ', 'Rec', 'LwQ', 'GLQ']
- Heating ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor']
- HeatingQC ['Ex', 'Gd', 'TA', 'Fa', 'Po']
- CentralAir ['Y', 'N']
- Electrical ['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix']
- KitchenQual ['Gd', 'TA', 'Ex', 'Fa']
- Functional ['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev']
- FireplaceQu ['NA', 'TA', 'Gd', 'Fa', 'Ex', 'Po']
- GarageType ['NA', 'Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', '2Types']
- GarageFinish ['NA', 'RFn', 'Unf', 'Fin']
- GarageQual ['NA', 'TA', 'Fa', 'Gd', 'Ex', 'Po']
- GarageCond ['NA', 'TA', 'Fa', 'Gd', 'Po', 'Ex']
- PavedDrive ['Y', 'N', 'P']
- PoolQC ['NA', 'Ex', 'Fa', 'Gd']
- Fence ['NA', 'MnPrv', 'GdWo', 'GdPrv', 'MnWw']
- MiscFeature ['NA', 'Shed', 'Gar2', 'Othr', 'TenC']
- SaleType ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth']
- SaleCondition ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family']
"""

DESC = {
    'boston': boston,
    'california': california,
    'adult': adult, 
    'titanic': titanic
}