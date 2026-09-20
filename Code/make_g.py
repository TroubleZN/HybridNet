import numpy as np

def make_treno():
    treno = np.zeros((1, 4, 89))  # 4 stages and 62 parameters

    ## Phenology
    treno[:, :, 0] = [0, 0, 0, 0.5]  # GDURootTemp, firt two are useless
    treno[:, :, 1] = [0, 0, 0, 53]  # firt two are useless, min temp GDU
    treno[:, :, 2] = [0, 0, 0, 91]  # firt two are useless, max temp GDU
    treno[:, :, 3] = [0, 0, 0, 2.9e6]  # firt two are useless, max light GPU, max data = 3.7e8
    treno[:, :, 4] = [0, 0, 0, 0.56]  # firt two are useless, power temp GPU
    treno[:, :, 5] = [0, 0, 0, 0.1]  # firt two are useless, power light GPU
    treno[:, :, 6] = [0, 100, 150, 50]  # additional GPU beyond previous stage
    id = 6

    ## Stress
    treno[:, :, id + 1] = 90  # best air temp
    treno[:, :, id + 2] = 20  # max air temp above best air temp
    treno[:, :, id + 3] = 35  # min air temp below best air temp
    treno[:, :, id + 4] = 2  # Air Heat Power
    treno[:, :, id + 5] = 2  # Air Cold Power
    treno[:, :, id + 6] = 80  # best root temp
    treno[:, :, id + 7] = 20  # max root temp above best air temp
    treno[:, :, id + 8] = 25  # min root temp below best air temp
    treno[:, :, id + 9] = 2  # root Heat Power
    treno[:, :, id + 10] = 2  # root Cold Power

    treno[:, :, id + 11] = 65  # best humidity
    treno[:, :, id + 12] = 20  # max humidity above best air temp
    treno[:, :, id + 13] = 25  # min humidity below best air temp
    treno[:, :, id + 14] = 2  # Air Wet Power
    treno[:, :, id + 15] = 2  # Air Dry Power
    treno[:, :, id + 16] = 2400000  # best light
    treno[:, :, id + 17] = 1000000  # max light above best light
    treno[:, :, id + 18] = 2000000  # min light below best light
    treno[:, :, id + 19] = 2  # coefficient for water absorption affected by plant density
    treno[:, :, id + 20] = 2  # coefficient for light absorption affected by plant density
    id = id + 20  # id = 22

    ## Water
    treno[:, :, id + 1] = 5  # ratio of xylem capacity over leaf weight
    treno[:, :, id + 2] = 5  # ratio of xylem capacity over stem weight
    treno[:, :, id + 3] = 0.1  # root water cap
    id = id + 3  # id = 25

    ## Transpiration
    treno[:, :, id + 1] = 0.2  # Temp Transpiration
    treno[:, :, id + 2] = 0.01  # Humidity Transpiration
    treno[:, :, id + 3] = 0.2  # Evap Transpiration 0
    treno[:, :, id + 4] = 1e3  # Evap Transpiration 1
    treno[:, :, id + 5] = 0.2  # Wind Transpiration 0
    treno[:, :, id + 6] = 300  # Wind Transpiration 1
    id = id + 6  # id = 31

    ## Biomass
    treno[:, :, id + 1] = 0.118  # photo per light *1e-6
    treno[:, :, id + 2] = 18.8  # photo per water
    treno[:, :, id + 3] = 0.01  # photowater per transpiration
    treno[:, :, id + 4] = 9.0  # photo per leaf
    treno[:, :, id + 5] = 22  # phloem per leaf
    treno[:, :, id + 6] = 2  # phloem per stem/leaf
    treno[:, :, id + 7] = 1  # dLeaf0
    treno[:, :, id + 8] = 0.1  # dLeaf1
    treno[:, :, id + 9] = 2  # dRoot0
    treno[:, :, id + 10] = 0.01  # dRoot1
    treno[:, :, id + 11] = 1  # dStem0
    treno[:, :, id + 12] = 0.01  # dStem1
    treno[:, :, id + 13] = 1.5  # dGrain0
    treno[:, :, id + 14] = 0.1  # dGrain1
    treno[:, :, id + 15] = 1.5  # dTassel0
    treno[:, :, id + 16] = 0.1  # dTassel1
    id = id + 16  # id = 47

    ## Maintenance and Senescense
    treno[:, :, id + 1] = 0.2  # LeafMS0
    treno[:, :, id + 2] = 0.001  # LeafMS1
    treno[:, :, id + 3] = 0.1  # RootMS0
    treno[:, :, id + 4] = 0.001  # RootMS1
    treno[:, :, id + 5] = 0.1  # StemMS0
    treno[:, :, id + 6] = 0.001  # StemMS1
    treno[:, :, id + 7] = 0.1  # GrainMS0
    treno[:, :, id + 8] = 0.001  # GrainMS1
    treno[:, :, id + 9] = 0.1  # TasselMS0
    treno[:, :, id + 10] = 0.001  # TasselMS1
    id = id + 10  # id = 57

    # Growth
    treno[:, :, id + 1] = 12  # Root density, gram per m
    treno[:, :, id + 2] = 18  # Stem density, gram per m
    treno[:, :, id + 3] = 15  # Tassel density, gram per m
    id = id + 3  # id = 60

    # HIR
    treno[:, :, id + 1] = 0.2  # HI Leaf
    treno[:, :, id + 2] = 0.2  # HI Stem
    treno[:, :, id + 3] = 0.1  # HI Root
    treno[:, :, id + 4] = 0.5  # HI Tassel
    treno[:, :, id + 5] = 0.5  # Air Stress Index Leaf
    treno[:, :, id + 6] = 0.5  # Air Stress Index Stem
    treno[:, :, id + 7] = 0.3  # Root Stress Index Root
    treno[:, :, id + 8] = 0.5  # Air Stress Index Tassel
    treno[:, :, id + 9] = 0.001  # HI Reference Threshold Leaf
    treno[:, :, id + 10] = 0.001  # HI Reference Threshold Stem
    treno[:, :, id + 11] = 0.001  # HI Reference Threshold Root
    treno[:, :, id + 12] = 0.001  # HI Reference Threshold Tassel
    treno[:, :, id + 13] = 1  # Maternal Health
    treno[:, :, id + 14] = 1  # Maternal Repair
    treno[:, :, id + 15] = 0.1  # LeafMS0
    treno[:, :, id + 16] = 0.001  # LeafMS1
    treno[:, :, id + 17] = 0.1  # RootMS0
    treno[:, :, id + 18] = 0.001  # RootMS1
    treno[:, :, id + 19] = 0.1  # StemMS0
    treno[:, :, id + 20] = 0.001  # StemMS1
    treno[:, :, id + 21] = 0.1  # TasselMS0
    treno[:, :, id + 22] = 0.001  # TasselMS1
    treno[:, :, id + 23] = 0.2  # HIBackgroundEffect
    treno[:, :, id + 24] = 1  # HIPotentialSensitivity
    id = id + 3  # id = 60


    Ltreno = np.zeros((1, 4, 89))  # 3 stages and 62 parameters

    ## Phenology
    Ltreno[:, :, 0] = [0, 0, 0, 0]  # GDURootTemp, firt two are useless
    Ltreno[:, :, 1] = [0, 0, 0, 40]  # firt two are useless, min temp GDU
    Ltreno[:, :, 2] = [0, 0, 0, 70]  # firt two are useless, max temp GDU
    Ltreno[:, :, 3] = [0, 0, 0, 2e6]  # firt two are useless, max light GPU, max data = 3.7e7
    Ltreno[:, :, 4] = [0, 0, 0, 0]  # firt two are useless, power temp GPU, power light GPU
    Ltreno[:, :, 5] = [0, 0, 0, 0]  # firt two are useless
    Ltreno[:, :, 6] = [0, 1, 1, 1]  # additional GPU beyond previous stage
    id = 6

    ## Stress
    Ltreno[:, :, id + 1] = 40  # best air temp
    Ltreno[:, :, id + 2] = 1  # max air temp above best air temp
    Ltreno[:, :, id + 3] = 1  # min air temp below best air temp
    Ltreno[:, :, id + 4] = 0  # Air Heat Power
    Ltreno[:, :, id + 5] = 0  # Air Cold Power
    Ltreno[:, :, id + 6] = 40  # best root temp
    Ltreno[:, :, id + 7] = 1  # max root temp above best air temp
    Ltreno[:, :, id + 8] = 1  # min root temp below best air temp
    Ltreno[:, :, id + 9] = 0  # root Heat Power
    Ltreno[:, :, id + 10] = 0  # root Cold Power

    Ltreno[:, :, id + 11] = 40  # best humidity
    Ltreno[:, :, id + 12] = 1  # max humidity above best air temp
    Ltreno[:, :, id + 13] = 1  # min humidity below best air temp
    Ltreno[:, :, id + 14] = 0  # Air Wet Power
    Ltreno[:, :, id + 15] = 0  # Air Dry Power
    Ltreno[:, :, id + 16] = 2e6  # best light
    Ltreno[:, :, id + 17] = 1000  # max light above best light
    Ltreno[:, :, id + 18] = 1000  # min light below best light
    Ltreno[:, :, id + 19] = 0  # coefficient for water absorption affected by plant density
    Ltreno[:, :, id + 20] = 0  # coefficient for light absorption affected by plant density
    id = id + 20  # id = 22

    ## Water
    Ltreno[:, :, id + 1] = 1  # ratio of xylem capacity over leaf weight
    Ltreno[:, :, id + 2] = 1  # ratio of xylem capacity over stem weight
    Ltreno[:, :, id + 3] = 1e-4  # root water cap
    id = id + 3  # id = 25

    ## Transpiration
    Ltreno[:, :, id + 1] = 0.01  # Temp Transpiration
    Ltreno[:, :, id + 2] = 1e-5  # Humidity Transpiration
    Ltreno[:, :, id + 3] = 0.001  # Evap Transpiration 0
    Ltreno[:, :, id + 4] = 1  # Evap Transpiration 1
    Ltreno[:, :, id + 5] = 0.001  # Wind Transpiration 0
    Ltreno[:, :, id + 6] = 0.001  # Wind Transpiration 1
    id = id + 6  # id = 31

    ## Biomass
    Ltreno[:, :, id + 1] = 1e-4  # photo per light *1e-6
    Ltreno[:, :, id + 2] = 0.01  # photo per water
    Ltreno[:, :, id + 3] = 1e-4  # photowater per transpiration
    Ltreno[:, :, id + 4] = 0.001  # photo per leaf
    Ltreno[:, :, id + 5] = 0.001  # phloem per leaf
    Ltreno[:, :, id + 6] = 0.2  # phloem per stem/leaf
    Ltreno[:, :, id + 7] = 1e-6  # dLeaf0
    Ltreno[:, :, id + 8] = 1e-6  # dLeaf1
    Ltreno[:, :, id + 9] = 1e-6  # dRoot0
    Ltreno[:, :, id + 10] = 1e-6  # dRoot1
    Ltreno[:, :, id + 11] = 1e-6  # dStem0
    Ltreno[:, :, id + 12] = 1e-6  # dStem1
    Ltreno[:, :, id + 13] = 1e-6  # dGrain0
    Ltreno[:, :, id + 14] = 1e-6  # dGrain1
    Ltreno[:, :, id + 15] = 1e-6  # dTassel0
    Ltreno[:, :, id + 16] = 1e-6  # dTassel1
    id = id + 16  # id = 47

    ## Maintenance and Senescense
    Ltreno[:, :, id + 1] = 1e-6  # LeafMS0
    Ltreno[:, :, id + 2] = 1e-6  # LeafMS1
    Ltreno[:, :, id + 3] = 1e-6  # RootMS0
    Ltreno[:, :, id + 4] = 1e-6  # RootMS1
    Ltreno[:, :, id + 5] = 1e-6  # StemMS0
    Ltreno[:, :, id + 6] = 1e-6  # StemMS1
    Ltreno[:, :, id + 7] = 1e-6  # GrainMS0
    Ltreno[:, :, id + 8] = 1e-6  # GrainMS1
    Ltreno[:, :, id + 9] = 1e-6  # GrainMS0
    Ltreno[:, :, id + 10] = 1e-6  # GrainMS1
    id = id + 10  # id = 57

    Ltreno[:, :, id + 1] = 0.1  # Root density, gram per m
    Ltreno[:, :, id + 2] = 0.1  # Stem density, gram per m
    Ltreno[:, :, id + 3] = 0.1  # Tassel density, gram per m
    id = id + 3  # id = 60

    # HIR
    Ltreno[:, :, id + 1] = 1e-6  # HI Leaf
    Ltreno[:, :, id + 2] = 1e-6  # HI Stem
    Ltreno[:, :, id + 3] = 1e-6  # HI Root
    Ltreno[:, :, id + 4] = 1e-6  # HI Tassel
    Ltreno[:, :, id + 5] = 1e-6  # Air Stress Index Leaf
    Ltreno[:, :, id + 6] = 1e-6  # Air Stress Index Stem
    Ltreno[:, :, id + 7] = 1e-6  # Root Stress Index Root
    Ltreno[:, :, id + 8] = 1e-6  # Air Stress Index Tassel
    Ltreno[:, :, id + 9] = 1e-6  # HI Reference Threshold Leaf
    Ltreno[:, :, id + 10] = 1e-6  # HI Reference Threshold Stem
    Ltreno[:, :, id + 11] = 1e-6  # HI Reference Threshold Root
    Ltreno[:, :, id + 12] = 1e-6  # HI Reference Threshold Tassel
    Ltreno[:, :, id + 13] = 0.8  # Maternal Health
    Ltreno[:, :, id + 14] = 0.8  # Maternal Repair
    Ltreno[:, :, id + 15] = 1e-6  # HILeafMS0
    Ltreno[:, :, id + 16] = 1e-6  # HILeafMS1
    Ltreno[:, :, id + 17] = 1e-6  # HIRootMS0
    Ltreno[:, :, id + 18] = 1e-6  # HIRootMS1
    Ltreno[:, :, id + 19] = 1e-6  # HIStemMS0
    Ltreno[:, :, id + 20] = 1e-6  # HIStemMS1
    Ltreno[:, :, id + 21] = 1e-6  # HITasselMS0
    Ltreno[:, :, id + 22] = 1e-6  # HITasselMS1
    Ltreno[:, :, id + 23] = 1e-6  # HIBackgroundEffect
    Ltreno[:, :, id + 24] = 1e-6  # HIPotentialSensitivity
    id = id + 14  # id = 74

    treno = np.maximum(treno, Ltreno)

    Utreno = np.zeros((1, 4, 89))  # 3 stages and 62 parameters

    ## Phenology
    Utreno[:, :, 0] = [0, 0, 0, 1]  # GDURootTemp, firt two are useless
    Utreno[:, :, 1] = [0, 0, 0, 60]  # firt two are useless, min temp GDU
    Utreno[:, :, 2] = [0, 0, 0, 110]  # firt two are useless, max temp GDU
    Utreno[:, :, 3] = [0, 0, 0, 4e6]  # firt two are useless, max light GPU, max data = 3.7e8
    Utreno[:, :, 4] = [0, 0, 0, 2]  # firt two are useless, power temp GPU, power light GPU
    Utreno[:, :, 5] = [0, 0, 0, 2]  # firt two are useless
    Utreno[:, :, 6] = [0, 500, 200, 150]  # additional GPU beyond previous stage
    id = 6

    ## Stress
    Utreno[:, :, id + 1] = 120  # best air temp
    Utreno[:, :, id + 2] = 50  # max air temp above best air temp
    Utreno[:, :, id + 3] = 50  # min air temp below best air temp
    Utreno[:, :, id + 4] = 20  # Air Heat Power
    Utreno[:, :, id + 5] = 20  # Air Cold Power
    Utreno[:, :, id + 6] = 120  # best root temp
    Utreno[:, :, id + 7] = 60  # max root temp above best air temp
    Utreno[:, :, id + 8] = 50  # min root temp below best air temp
    Utreno[:, :, id + 9] = 20  # root Heat Power
    Utreno[:, :, id + 10] = 20  # root Cold Power

    Utreno[:, :, id + 11] = 80  # best humidity
    Utreno[:, :, id + 12] = 40  # max humidity above best air temp
    Utreno[:, :, id + 13] = 40  # min humidity below best air temp
    Utreno[:, :, id + 14] = 20  # Air Wet Power
    Utreno[:, :, id + 15] = 20  # Air Dry Power
    Utreno[:, :, id + 16] = 3.5e6  # best light
    Utreno[:, :, id + 17] = 2e6  # max light above best light
    Utreno[:, :, id + 18] = 2e6  # min light below best light
    Utreno[:, :, id + 19] = 20  # coefficient for water absorption affected by plant density
    Utreno[:, :, id + 20] = 20  # coefficient for light absorption affected by plant density
    id = id + 20  # id = 22

    ## Water
    Utreno[:, :, id + 1] = 50  # ratio of xylem capacity over leaf weight
    Utreno[:, :, id + 2] = 50  # ratio of xylem capacity over stem weight
    Utreno[:, :, id + 3] = 1  # root water cap
    id = id + 3  # id = 25

    ## Transpiration
    Utreno[:, :, id + 1] = 0.8  # Temp Transpiration
    Utreno[:, :, id + 2] = 0.2  # Humidity Transpiration
    Utreno[:, :, id + 3] = 1  # Evap Transpiration 0
    Utreno[:, :, id + 4] = 1e6  # Evap Transpiration 1
    Utreno[:, :, id + 5] = 1  # Wind Transpiration 0
    Utreno[:, :, id + 6] = 1000  # Wind Transpiration 1
    id = id + 6  # id = 31

    ## Biomass
    Utreno[:, :, id + 1] = 118  # photo per light *1e-6
    Utreno[:, :, id + 2] = 1380  # photo per water
    Utreno[:, :, id + 3] = 0.5  # photowater per transpiration
    Utreno[:, :, id + 4] = 100  # photo per leaf
    Utreno[:, :, id + 5] = 100  # phloem per leaf
    Utreno[:, :, id + 6] = 5  # phloem per stem/leaf
    Utreno[:, :, id + 7] = 10  # dLeaf0
    Utreno[:, :, id + 8] = 0.5  # dLeaf1
    Utreno[:, :, id + 9] = 10  # dRoot0
    Utreno[:, :, id + 10] = 0.5  # dRoot1
    Utreno[:, :, id + 11] = 10  # dStem0
    Utreno[:, :, id + 12] = 0.5  # dStem1
    Utreno[:, :, id + 13] = [0, 0, 5, 10]  # dGrain0
    Utreno[:, :, id + 14] = [0, 1, 1, 1]  # dGrain1
    Utreno[:, :, id + 15] = [0, 5, 2, 1e-6]  # dTassel0
    Utreno[:, :, id + 16] = [0, 1, 1, 1]  # dTassel1
    id = id + 16  # id = 47

    ## Maintenance and Senescense
    Utreno[:, :, id + 1] = 0.5  # LeafMS0
    Utreno[:, :, id + 2] = 0.5  # LeafMS1
    Utreno[:, :, id + 3] = 0.5  # RootMS0
    Utreno[:, :, id + 4] = 0.5  # RootMS1
    Utreno[:, :, id + 5] = 0.5  # StemMS0
    Utreno[:, :, id + 6] = 0.5  # StemMS1
    Utreno[:, :, id + 7] = [0, 0, 0.5, 0.5]  # GrainMS0
    Utreno[:, :, id + 8] = [0, 0, 0.5, 0.5]  # GrainMS1
    Utreno[:, :, id + 9] = [0, 0.5, 0.5, 0.5]  # TasselMS0
    Utreno[:, :, id + 10] = [0, 0.5, 0.5, 0.5]  # TasselMS1
    id = id + 10  # id = 57

    Utreno[:, :, id + 1] = 100  # Root density, gram per m
    Utreno[:, :, id + 2] = 100  # Stem density, gram per m
    Utreno[:, :, id + 3] = 100  # Tassel density, gram per m
    id = id + 3  # id = 60

    # HIR
    Utreno[:, :, id + 1] = 1  # HI Leaf
    Utreno[:, :, id + 2] = 1  # HI Stem
    Utreno[:, :, id + 3] = 1  # HI Root
    Utreno[:, :, id + 4] = 1  # HI Tassel
    Utreno[:, :, id + 5] = 100  # Air Stress Index Leaf
    Utreno[:, :, id + 6] = 100  # Air Stress Index Stem
    Utreno[:, :, id + 7] = 100  # Root Stress Index Root
    Utreno[:, :, id + 8] = 100  # Air Stress Index Tassel
    Utreno[:, :, id + 9] = 100  # HI Reference Threshold Leaf
    Utreno[:, :, id + 10] = 100  # HI Reference Threshold Stem
    Utreno[:, :, id + 11] = 100  # HI Reference Threshold Root
    Utreno[:, :, id + 12] = 100  # HI Reference Threshold Tassel
    Utreno[:, :, id + 13] = 1  # Maternal Health
    Utreno[:, :, id + 14] = 1  # Maternal Repair
    Utreno[:, :, id + 15] = 0.5  # HILeafMS0
    Utreno[:, :, id + 16] = 0.5  # HILeafMS1
    Utreno[:, :, id + 17] = 0.5  # HIRootMS0
    Utreno[:, :, id + 18] = 0.5  # HIRootMS1
    Utreno[:, :, id + 19] = 0.5  # HIStemMS0
    Utreno[:, :, id + 20] = 0.5  # HIStemMS1
    Utreno[:, :, id + 21] = [0, 0.5, 0.5, 0.5]  # HITasselMS0
    Utreno[:, :, id + 22] = [0, 0.5, 0.5, 0.5]  # HITasselMS1
    Utreno[:, :, id + 23] = 0.5  # HIBackgroundEffect
    Utreno[:, :, id + 24] = 2  # HIPotentialSensitivity
    id = id + 14  # id = 74

    treno = np.minimum(treno, Utreno)


    #
    # dtreno = np.zeros((1, 3, 51))  # 3 stages and 62 parameters
    #
    # ## Phenology
    # dtreno[:, :, 0] = [0, 0, 1e-4]  # GDURootTemp, firt two are useless
    # dtreno[:, :, 1] = [0, 0, 1e-3]  # firt two are useless, min temp GDU
    # dtreno[:, :, 2] = [0, 0, 1e-3]  # firt two are useless, max temp GDU
    # dtreno[:, :, 3] = [0, 0, 2e3]  # firt two are useless, max light GPU, max data = 3.7e8
    # dtreno[:, :, 4] = [0, 0, 1e-4]  # firt two are useless, power temp GPU, power light GPU
    # dtreno[:, :, 5] = [0, 0, 1e-4]  # firt two are useless
    # dtreno[:, :, 6] = [0, 1, 1]  # additional GPU beyond previous stage
    # id = 6
    #
    # ## Stress
    # dtreno[:, :, id + 1] = 0.01  # best air temp
    # dtreno[:, :, id + 2] = 0.01  # max air temp above best air temp
    # dtreno[:, :, id + 3] = 0.01  # min air temp below best air temp
    # dtreno[:, :, id + 4] = 1e-4  # Air Heat Power
    # dtreno[:, :, id + 5] = 1e-4  # Air Cold Power
    # dtreno[:, :, id + 6] = 0.01  # best root temp
    # dtreno[:, :, id + 7] = 0.01  # max root temp above best air temp
    # dtreno[:, :, id + 8] = 0.01  # min root temp below best air temp
    # dtreno[:, :, id + 9] = 1e-4  # root Heat Power
    # dtreno[:, :, id + 10] = 1e-4  # root Cold Power
    # id = id + 10  # id = 17
    #
    # ## Water
    # dtreno[:, :, id + 1] = 0.01  # ratio of xylem capacity over leaf weight
    # dtreno[:, :, id + 2] = 0.01  # ratio of xylem capacity over stem weight
    # dtreno[:, :, id + 3] = 1e-6  # root water cap
    # id = id + 3  # id = 20
    #
    # ## Transpiration
    # dtreno[:, :, id + 1] = 1e-4  # Temp Transpiration
    # dtreno[:, :, id + 2] = 1e-4  # Humidity Transpiration
    # dtreno[:, :, id + 3] = 1e-4  # Evap Transpiration 0
    # dtreno[:, :, id + 4] = 1  # Evap Transpiration 1
    # dtreno[:, :, id + 5] = 1e-4  # Wind Transpiration 0
    # dtreno[:, :, id + 6] = 1e-3  # Wind Transpiration 1
    # id = id + 6  # id = 26
    #
    # ## Biomass
    # dtreno[:, :, id + 1] = 1e-4  # photo per light *1e-6
    # dtreno[:, :, id + 2] = 1e-4  # photo per water
    # dtreno[:, :, id + 3] = 1e-4  # photowater per transpiration
    # dtreno[:, :, id + 4] = 1e-4  # photo per leaf
    # dtreno[:, :, id + 5] = 1e-4  # phloem per leaf
    # dtreno[:, :, id + 6] = 1e-4  # phloem per stem/leaf
    # dtreno[:, :, id + 7] = 1e-4  # dLeaf0
    # dtreno[:, :, id + 8] = 1e-4  # dLeaf1
    # dtreno[:, :, id + 9] = 1e-4  # dRoot0
    # dtreno[:, :, id + 10] = 1e-4  # dRoot1
    # dtreno[:, :, id + 11] = 1e-4  # dStem0
    # dtreno[:, :, id + 12] = 1e-4  # dStem1
    # dtreno[:, :, id + 13] = 1e-4  # dGrain0
    # dtreno[:, :, id + 14] = 1e-4  # dGrain1
    # id = id + 14  # id = 40
    #
    # ## Maintenance and Senescense
    # dtreno[:, :, id + 1] = 1e-4  # LeafMS0
    # dtreno[:, :, id + 2] = 1e-4  # LeafMS1
    # dtreno[:, :, id + 3] = 1e-4  # RootMS0
    # dtreno[:, :, id + 4] = 1e-4  # RootMS1
    # dtreno[:, :, id + 5] = 1e-4  # StemMS0
    # dtreno[:, :, id + 6] = 1e-4  # StemMS1
    # dtreno[:, :, id + 7] = 1e-4  # GrainMS0
    # dtreno[:, :, id + 8] = 1e-4  # GrainMS1
    # id = id + 8  # id = 48
    #
    # dtreno[:, :, id + 1] = 1e-4  # Root density, gram per m
    # dtreno[:, :, id + 2] = 1e-4  # Stem density, gram per m
    # dtreno[:, :, id + 3] = [1e-4, 1e-4, 1e-4]  # lodging effect for 1, 2, 3, 4, 5 are 1, 1, m1, m1*m2, and m1*m2*m3. Not trneo property, just treated together for convenience. This parameter is only tuned for all hybrids. When tuning individual hybrids, L and U will be fixed the same.

    # dtreno = (Utreno-Ltreno)/1000

    return treno, Utreno, Ltreno
