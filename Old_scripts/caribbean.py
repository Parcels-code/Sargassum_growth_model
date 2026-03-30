SEA = """Caribbean Sea	21.60000	-87.07500	250001006	1
Caribbean Sea	21.86700	-84.95000	250001006	2
Caribbean Sea	22.81400	-82.58600	250001006	3
Caribbean Sea	22.81400	-81.12000	250001006	4
Caribbean Sea	20.06700	-74.29600	250001006	5
Caribbean Sea	19.75000	-73.42100	250001006	6
Caribbean Sea	18.60800	-68.32500	250001006	7
Caribbean Sea	18.47900	-67.16700	250001006	8
Caribbean Sea	18.38200	-65.64400	250001006	9
Caribbean Sea	18.38200	-62.19000	250001006	10
Caribbean Sea	13.25000	-59.12000	250001006	11
Caribbean Sea	10.83600	-60.90800	250001006	12
Caribbean Sea	10.13700	-60.99200	250001006	13
Caribbean Sea	9.51700	-60.96700	250001006	14
Caribbean Sea	8.65000	-62.46000	250001006	15
Caribbean Sea	7.42000	-76.37000	250001006	16
Caribbean Sea	8.00000	-77.70000	250001006	17
Caribbean Sea	9.15000	-78.40000	250001006	18
Caribbean Sea	9.15000	-79.50000	250001006	19
Caribbean Sea	8.20000	-80.70000	250001006	20
Caribbean Sea	8.61000	-82.27000	250001006	21
Caribbean Sea	14.00000	-86.00000	250001006	22
Caribbean Sea	15.00000	-90.00000	250001006	23
Caribbean Sea	21.60000	-87.07500	250001006	24"""

def get_sea_points():
    points = []
    for line in SEA.splitlines():
        _, lat, lon, _, _ = line.split("\t")
        points.append((float(lon), float(lat)))
    return points
get_sea_points()
