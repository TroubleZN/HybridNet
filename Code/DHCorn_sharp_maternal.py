import numpy as np
import torch
from torch import tensor, pow, sum, min, fmin, max, fmax, mean, exp
import pandas as pd
from time import time


def safe_pow(base, exponent):
    return torch.pow(base.clamp(min=1e-8), exponent)

def smooth_step(x, center, sharpness):
    return torch.sigmoid((x - center) * sharpness)


def dhcorn(envi, mg, geno, maternal_effect=None, pheno=0, device='cpu', stage_sharpness=0.12):
    # ph.Yield, nsamples by 1
    # ph.LeafDry, nsamples by 130
    # ph.LeafWet, nsamples by 130
    # ph.StemDry, nsamples by 130
    # ph.StemWet, nsamples by 130
    # ph.Height, nsamples by 130
    # ph.Tiller, nsamples by 130

    # envi is nsamples by 4320 hours by 13 variables
    # mg is nsamples by 2 (1st is harvesting dates and 2nd is density)
    # geno is nsamples by 3 stages by 51 parameters
    # nD = nH / 24
    # nT = size(geno, 2) # number of transition points

    envi = envi.to(device).to(torch.float)
    geno = geno.to(device).to(torch.float)
    mg = mg.to(device).to(torch.float)
    if maternal_effect is not None:
        maternal_effect = maternal_effect.to(device).to(torch.float)

    nsamples, _, _ = envi.size()  # number of samples and number of hours
    start = time()

    # Environment data
    E_AirTemp = envi[:, :, 0]  # mean = 56
    E_SoilTemp = envi[:, :, 1:5]  # 4 layers, mean = [59, 56, 55, 52]
    E_SoilMoist = envi[:, :, 5:9]  # 4 layers, mean = [30, 29, 21, 25]
    E_Humidity = envi[:, :, 9]  # mean = 6.7
    E_Light = envi[:, :, 10]  # mean = 4e7
    E_Wind = envi[:, :, 11]  # mean = 4.8
    E_Evap = envi[:, :, 12]  # mean = 5.6e-3, min = -2.5e-3, max = 0.04

    # Management data
    M_Harvest = mg[:, 0]  # second day is always planting date
    M_Density = mg[:, 1]

    # Phenology
    g_GDURootTemp = geno[:, -1, 0]
    g_MinTemp = geno[:, -1, 1]
    g_MaxTemp = geno[:, -1, 2]
    g_MaxLight = geno[:, -1, 3]
    g_GPUTemp = geno[:, -1, 4]
    g_GPULight = geno[:, -1, 5]
    g_Stage = torch.cumsum(geno[:, :, 6], 1)
    id = 6

    # Stress
    g_BestAirTemp = geno[:, :, id + 1]
    g_MaxAirTemp = g_BestAirTemp + geno[:, :, id + 2]
    g_MinAirTemp = g_BestAirTemp - geno[:, :, id + 3]
    g_AirHeatPower = geno[:, :, id + 4]
    g_AirColdPower = geno[:, :, id + 5]

    g_BestRootTemp = geno[:, :, id + 6]
    g_MaxRootTemp = g_BestRootTemp + geno[:, :, id + 7]
    g_MinRootTemp = g_BestRootTemp - geno[:, :, id + 8]
    g_RootHeatPower = geno[:, :, id + 9]
    g_RootColdPower = geno[:, :, id + 10]

    g_BestHumidity = geno[:, :, id + 11]
    g_MaxHumidity = g_BestHumidity + geno[:, :, id + 12]
    g_MinHumidity = g_BestHumidity - geno[:, :, id + 13]
    g_AirWetPower = geno[:, :, id + 14]
    g_AirDryPower = geno[:, :, id + 15]

    g_BestLight = geno[:, :, id + 16]
    g_MaxLightS = g_BestLight + geno[:, :, id + 17]
    g_MinLightS = g_BestLight - geno[:, :, id + 18]
    g_RadiationPower = geno[:, :, id + 19]
    g_ShadingPower = geno[:, :, id + 20]

    # g_AccuAirStressRecovery = geno[:, :, id + 11]
    # g_AccuRootStressRecovery = geno[:, :, id + 12]
    # g_AccuAirStress = geno[:, :, id + 13]
    # g_AccuRootStress = geno[:, :, id + 14]
    # g_WaterPlantDensity = geno[:, :, id + 15]
    # g_LightPlantDensity = geno[:, :, id + 16]
    id = id + 20  # id = 26

    # Water
    g_Xylem_Leaf = geno[:, :, id + 1]
    g_Xylem_Stem = geno[:, :, id + 2]
    g_RootWaterCap = geno[:, :, id + 3]
    id = id + 3  # id = 29

    # Transpiration
    g_TempTranspiration = geno[:, :, id + 1]
    g_HumidityTranspiration = geno[:, :, id + 2]
    g_EvapTranspiration0 = geno[:, :, id + 3]
    g_EvapTranspiration1 = geno[:, :, id + 4]
    g_WindTranspiration0 = geno[:, :, id + 5]
    g_WindTranspiration1 = geno[:, :, id + 6]
    id = id + 6  # id = 35

    # Biomass
    g_Photo_Light = geno[:, :, id + 1] * 1e-6
    g_Photo_Water = geno[:, :, id + 2]
    g_PhotoWater_Transpiration = geno[:, :, id + 3]
    g_Photo_Leaf = geno[:, :, id + 4]
    g_Phloem_Leaf = geno[:, :, id + 5]
    g_Phloem_Stem = geno[:, :, id + 5] * geno[:, :, id + 6]
    g_dLeaf0 = geno[:, :, id + 7]
    g_dLeaf1 = geno[:, :, id + 8]
    g_dRoot0 = geno[:, :, id + 9]
    g_dRoot1 = geno[:, :, id + 10]
    g_dStem0 = geno[:, :, id + 11]
    g_dStem1 = geno[:, :, id + 12]
    g_dGrain0 = geno[:, :, id + 13]
    g_dGrain1 = geno[:, :, id + 14]
    g_dTassel0 = geno[:, :, id + 15]
    g_dTassel1 = geno[:, :, id + 16]

    id = id + 16  # id = 51

    # Maintenance and Senescense
    g_LeafMS0 = geno[:, :, id + 1]
    g_LeafMS1 = geno[:, :, id + 2]
    g_RootMS0 = geno[:, :, id + 3]
    g_RootMS1 = geno[:, :, id + 4]
    g_StemMS0 = geno[:, :, id + 5]
    g_StemMS1 = geno[:, :, id + 6]
    g_GrainMS0 = geno[:, :, id + 7]
    g_GrainMS1 = geno[:, :, id + 8]
    g_TasselMS0 = geno[:, :, id + 9]
    g_TasselMS1 = geno[:, :, id + 10]
    id = id + 10  # id = 61

    # Growth
    g_RootDensity = geno[:, :, id + 1]
    g_StemDensity = geno[:, :, id + 2]
    g_TasselDensity = geno[:, :, id + 3]

    id = id + 3  # id = 64

    # HI
    g_HI_Leaf = geno[:, :, id + 1]
    g_HI_Stem = geno[:, :, id + 2]
    g_HI_Root = geno[:, :, id + 3]
    g_HI_Tassel = geno[:, :, id + 4]
    g_AirStressIndexLeaf = geno[:, :, id + 5]
    g_AirStressIndexStem = geno[:, :, id + 6]
    g_AirStressIndexRoot = geno[:, :, id + 7]
    g_AirStressIndexTassel = geno[:, :, id + 8]
    g_HIReferenceThresholdLeaf = geno[:, :, id + 9]
    g_HIReferenceThresholdStem = geno[:, :, id + 10]
    g_HIReferenceThresholdRoot = geno[:, :, id + 11]
    g_HIReferenceThresholdTassel = geno[:, :, id + 12]
    g_MaternalHealth = geno[:, :, id + 13]
    g_MaternalRepair = geno[:, :, id + 14]
    g_HILeafMS0 = geno[:, :, id + 15]
    g_HILeafMS1 = geno[:, :, id + 16]
    g_HIStemMS0 = geno[:, :, id + 17]
    g_HIStemMS1 = geno[:, :, id + 18]
    g_HITasselMS0 = geno[:, :, id + 19]
    g_HITasselMS1 = geno[:, :, id + 20]
    g_HIRootMS0 = geno[:, :, id + 21]
    g_HIRootMS1 = geno[:, :, id + 22]
    g_start0 = geno[:, -1, id + 23]
    g_start1 = geno[:, -1, id + 24]
    id = id + 24  # id = 86

    # Initialization of duration variables
    # v_ARootTemp = torch.zeros(nsamples, 24).to(device)  # (1)
    # v_GDUTemp = torch.zeros(nsamples, 24).to(device)  # (2)
    # v_GPUTemp = torch.zeros(nsamples, 24).to(device)  # (3)
    # v_GPULight = torch.zeros(nsamples, 24).to(device)  # (4)
    v_GPU = torch.zeros(nsamples, 130).to(device)  # (5)
    ts_all = torch.zeros(nsamples, 4, 130).to(device)  # (6)

    # v_AirHeat = torch.zeros(nsamples).to(device)  # (7)
    # v_AirCold = torch.zeros(nsamples).to(device)  # (8)
    # v_RootHeat = torch.zeros(nsamples).to(device)  # (9)
    # v_RootCold = torch.zeros(nsamples).to(device)  # (10)
    # v_AirStress = torch.zeros(nsamples).to(device)  # (11)
    # v_RootStress = torch.zeros(nsamples).to(device)  # (12)
    v_AccuAirStress = torch.zeros(nsamples, 130).to(device)
    v_AccuRootStress = torch.zeros(nsamples, 130).to(device)

    # v_MXylemCap = torch.zeros(nsamples).to(device)  # (13)
    # v_TXylemCap = torch.zeros(nsamples).to(device)  # (14)
    # v_RootWaterCap = torch.zeros(nsamples).to(device)  # (15)
    # v_MWaterUptakeCap = torch.zeros(nsamples).to(device)  # (16)
    # v_TWaterUptakeCap = torch.zeros(nsamples).to(device)  # (17)
    # v_TotalWaterUptake = torch.zeros(nsamples).to(device)  # (18)
    # v_MWaterUptake = torch.zeros(nsamples).to(device)  # (19)
    # v_TWaterUptake = torch.zeros(nsamples).to(device)  # (20)
    v_MWater = torch.zeros(nsamples, 130).to(device)  # (21)

    # v_TempTranspiration = torch.zeros(nsamples).to(device)  # (23)
    # v_HumidityTranspiration = torch.zeros(nsamples).to(device)  # (24)
    # v_EvapTranspiration = torch.zeros(nsamples).to(device)  # (25)
    # v_WindTranspiration = torch.zeros(nsamples).to(device)  # (26)
    v_MTranspiration = torch.zeros(nsamples).to(device)  # (27)
    # v_PhotoLightCap = torch.zeros(nsamples).to(device)  # (29)
    # v_MPhotoWaterCap = torch.zeros(nsamples).to(device)  # (30)
    # v_MPhotoLeafCap = torch.zeros(nsamples).to(device)  # (32)
    # v_MPhloemCap = torch.zeros(nsamples).to(device)  # (34)
    # v_MPhoto = torch.zeros(nsamples).to(device)  # (36)
    # v_MLeafGrowthCap = torch.zeros(nsamples).to(device)  # (38)
    # v_MRootGrowthCap = torch.zeros(nsamples).to(device)  # (39)
    # v_MStemGrowthCap = torch.zeros(nsamples).to(device)  # (40)
    # v_MGrainGrowthCap = torch.zeros(nsamples).to(device)  # (41)
    # v_MGrowthCap = torch.zeros(nsamples).to(device)  # (42)
    # v_MGrowth = torch.zeros(nsamples).to(device)  # (48)
    v_MBiomass = torch.zeros(nsamples, 130).to(device)  # (50)
    v_MWaterPhoto = torch.zeros(nsamples).to(device)  # (52)

    v_MLeafMS = torch.zeros(nsamples).to(device)  # (54)
    v_MRootMS = torch.zeros(nsamples).to(device)  # (55)
    v_MStemMS = torch.zeros(nsamples).to(device)  # (56)
    v_MGrainMS = torch.zeros(nsamples).to(device)  # (57)
    v_TasselMS = torch.zeros(nsamples).to(device)  # (57)

    v_MLeafWeight = 0.1 * torch.ones(nsamples, 130).to(device)  # initial 1 gram (62)
    v_MRootWeight = 0.1 * torch.ones(nsamples, 130).to(device)  # initial 1 gram (63)
    v_MRootLength = 0.1 * torch.ones(nsamples, 130).to(device) / mean(g_RootDensity, 1).view(nsamples, 1).to(
        device)  # (64)
    v_MRootLayerLength = torch.zeros(nsamples, 130, 4).to(device)
    v_MRootLayerLength[:, 0, 0] = 0.1 / mean(g_RootDensity, 1).to(device)  # (65)
    v_MStemWeight = 0.1 * torch.ones(nsamples, 130).to(device)  # (66)
    v_MStemHeight = 0.1 * torch.ones(nsamples, 130).to(device) / mean(g_StemDensity, 1).view(nsamples, 1).to(
        device)  # (67)
    v_MStemHeight[:, 1:] = torch.nan
    v_MGrainWeight = torch.zeros(nsamples, 130).to(device)  # (68)
    v_TasselHeight = torch.zeros(nsamples, 130).to(device)  # (68)
    v_TasselHeight[:, 1:] = torch.nan
    v_TasselWeight = torch.zeros(nsamples, 130).to(device)  # (68)
    v_MShootWeight = v_MLeafWeight + v_MStemWeight + v_MGrainWeight + v_TasselWeight  # (69)

    v_HIMatterCapLeaf = torch.zeros(nsamples, 130).to(device)
    v_HIMatterCapStem = torch.zeros(nsamples, 130).to(device)
    v_HIMatterCapRoot = torch.zeros(nsamples, 130).to(device)
    v_HIMatterCapTassel = torch.zeros(nsamples, 130).to(device)

    v_HIRate = torch.zeros(nsamples, 130).to(device)
    v_HIRate[:, 1:] = torch.nan
    v_HIPotential = torch.zeros(nsamples, 130).to(device)

    ph = {}

    # print(time()-start)

    start = time()
    for d in range(1, int(max(M_Harvest)) + 1):
        # Phenology

        v_ARootTemp = sum(v_MRootLayerLength[:, [d - 1], :].repeat(1, 24, 1) * E_SoilTemp[:, 24 * d:24 * d + 24, :], 2) \
                      / sum(v_MRootLayerLength[:, [d - 1], :].repeat(1, 24, 1),
                            2)  # nsamples by 24. (1)

        v_GDUTemp = min(max((1 - g_GDURootTemp.view(nsamples, 1).repeat(1, 24)) * E_AirTemp[:, 24 * d:24 * d + 24]
                            + g_GDURootTemp.view(nsamples, 1).repeat(1, 24) * v_ARootTemp,
                            g_MinTemp.view(nsamples, 1).repeat(1, 24)), g_MaxTemp.view(nsamples, 1)) \
                    - g_MinTemp.view(nsamples, 1)  # nsamples by 24. (2)

        v_GPUTemp = max(v_GDUTemp / (g_MaxTemp - g_MinTemp).view(nsamples, 1).repeat(1, 24), tensor(1e-16))  # nsamples by 24. (3)

        v_GPULight = max(min(E_Light[:, 24 * d:24 * d + 24] / g_MaxLight.view(nsamples, 1).repeat(1, 24), tensor(1)), tensor(1e-16))  # nsamples by 24. (4)

        v_GPU[:, d] = v_GPU[:, d - 1].clone() + sum(pow(v_GPUTemp, g_GPUTemp.view(nsamples, 1)) * pow(v_GPULight, g_GPULight.view(nsamples, 1)), 1)  # nsamples by 1. (5). feedback on g_Stage

        v_gpu_d = v_GPU[:, d].clone()
        gate_12 = smooth_step(v_gpu_d, g_Stage[:, 1], stage_sharpness)
        gate_23 = smooth_step(v_gpu_d, g_Stage[:, 2], stage_sharpness)
        gate_34 = smooth_step(v_gpu_d, g_Stage[:, 3], stage_sharpness)

        ts_all[:, 0, d] = 1 - gate_12
        ts_all[:, 1, d] = gate_12 * (1 - gate_23)
        ts_all[:, 2, d] = gate_23 * (1 - gate_34)
        ts_all[:, 3, d] = gate_34
        ts = ts_all[:, :, d].clone()

        # Stress
        temp1 = E_AirTemp[:, 24 * d:24 * d + 24] - sum(g_BestAirTemp * ts, 1).view(nsamples, 1).repeat(1, 24)

        v_AirHeat = mean(
            safe_pow(min(max(temp1, tensor(0)) / sum((g_MaxAirTemp - g_BestAirTemp) * ts, 1).view(nsamples, 1).repeat(1, 24),
                    tensor(1)),
                sum(g_AirHeatPower * ts, 1).view(nsamples, 1).repeat(1, 24)), 1)

        v_AirCold = mean(
            safe_pow(min(min(temp1, tensor(0)) / sum((g_MinAirTemp - g_BestAirTemp) * ts, 1).view(nsamples, 1).repeat(1, 24),
                    tensor(1)),
                sum(g_AirColdPower * ts, 1).view(nsamples, 1).repeat(1, 24)), 1)

        temp2 = v_ARootTemp - sum(g_BestRootTemp * ts, 1).view(nsamples, 1)
        temp3 = sum(g_RootHeatPower * ts, 1).view(nsamples, 1)
        temp3_2 = sum((g_MaxRootTemp - g_BestRootTemp) * ts, 1).view(nsamples, 1)
        temp4 = sum(g_RootColdPower * ts, 1).view(nsamples, 1)
        temp4_2 = sum((g_MinRootTemp - g_BestRootTemp) * ts, 1).view(nsamples, 1)

        v_RootHeat = mean(safe_pow(min(max(temp2, tensor(0)) / temp3_2, tensor(1)), temp3), 1)
        v_RootCold = mean(safe_pow(min(min(temp2, tensor(0)) / temp4_2, tensor(1)), temp4), 1)

        v_AirStress = 1 - (1 - v_AirHeat) * (1 - v_AirCold)  # nsamples by 1, (11)

        v_RootStress = 1 - (1 - v_RootHeat) * (1 - v_RootCold)  # nsamples by 1, (12)


        humid = E_Humidity[:, 24 * d:24 * d + 24] - sum(g_BestHumidity * ts, 1).view(nsamples, 1).repeat(1, 24)

        v_AirWet = mean(
            safe_pow(min(max(humid, tensor(0)) / sum((g_MaxHumidity - g_BestHumidity) * ts, 1).view(nsamples, 1).repeat(1, 24),
                    tensor(1)),
                sum(g_AirWetPower * ts, 1).view(nsamples, 1).repeat(1, 24)), 1)

        v_AirDry = mean(
            safe_pow(min(min(humid, tensor(0)) / sum((g_MinHumidity - g_BestHumidity) * ts, 1).view(nsamples, 1).repeat(1, 24),
                    tensor(1)),
                sum(g_AirDryPower * ts, 1).view(nsamples, 1).repeat(1, 24)), 1)

        v_HumidityStress = 1 - (1 - v_AirDry) * (1 - v_AirWet)

        light = E_Light[:, 24 * d:24 * d + 24] - sum(g_BestLight * ts, 1).view(nsamples, 1).repeat(1, 24)

        v_RadiationStress = mean(
            safe_pow(min(max(light, tensor(0)) / sum((g_MaxLightS - g_BestLight) * ts, 1).view(nsamples, 1).repeat(1, 24),
                    tensor(1)),
                sum(g_RadiationPower * ts, 1).view(nsamples, 1).repeat(1, 24)), 1)

        v_ShadingStress = mean(
            safe_pow(min(min(light, tensor(0)) / sum((g_MinLightS- g_BestLight) * ts, 1).view(nsamples, 1).repeat(1, 24),
                    tensor(1)),
                sum(g_ShadingPower * ts, 1).view(nsamples, 1).repeat(1, 24)), 1)

        v_LightStress = 1 - (1 - v_RadiationStress) * (1 - v_ShadingStress)

        # v_AccuAirStress[:, d] = max(v_AccuAirStress[:, d-1].clone() - v_AccuAirStress[:, d-1].clone() * v_MShootWeight[:, d-1].clone() / sum(g_AccuAirStressRecovery * ts, 1) + v_AirStress, tensor(0))
        #
        # v_AccuRootStress[:, d] = max(v_AccuRootStress[:, d-1].clone() - v_AccuRootStress[:, d-1].clone() * v_MRootWeight[:, d-1].clone() / sum(g_AccuRootStressRecovery * ts, 1) + v_RootStress, tensor(0))
        #
        # v_AccuAirStressIndex = min(max(1 - v_AccuAirStress[:, d].clone() * (v_AccuAirStress[:, d].clone() + sum(g_AccuAirStress * ts, 1)), tensor(0)), tensor(1))
        # v_AccuRootStressIndex = min(max(1 - v_AccuRootStress[:, d].clone() * (v_AccuRootStress[:, d].clone() + sum(g_AccuRootStress * ts, 1)), tensor(0)), tensor(1))
        #
        # v_WaterDensityIndex = exp(- sum(g_WaterPlantDensity * ts, 1) * M_Density + sum(g_WaterPlantDensity * ts, 1))
        # v_LightDensityIndex = exp(- sum(g_LightPlantDensity * ts, 1) * M_Density + sum(g_LightPlantDensity * ts, 1))

        # Water

        v_MXylemCap = sum(g_Xylem_Leaf * ts, 1) * v_MLeafWeight[:, d - 1].clone() \
                      + sum(g_Xylem_Stem * ts, 1) * v_MStemWeight[:, d - 1].clone()  # nsamples by 1, (13)

        temp6_1 = 1 - v_RootStress
        temp6_2 = sum(g_RootWaterCap * ts, 1)
        temp6_3 = v_MRootLayerLength[:, d - 1, :]
        temp6_4 = sum(E_SoilMoist[:, 24 * d:24 * d + 24, :], 1)
        v_RootWaterCap = temp6_1 * temp6_2 / M_Density * sum(temp6_3 * temp6_4, 1)  # nsamples by 1, (15)

        v_MWaterUptakeCap = max(v_MXylemCap - (v_MWater[:, d - 1] - v_MTranspiration - v_MWaterPhoto),
                                tensor(0))  # nsamples by 1, (16)

        v_TotalWaterUptake = min(v_MWaterUptakeCap, v_RootWaterCap)  # nsamples by 1, (18)

        v_MWaterUptake = min(v_MWaterUptakeCap, v_TotalWaterUptake)  # nsamples by 1, (19)

        v_MWater = v_MWater.clone()
        v_MWater[:, d] = min(v_MXylemCap,
                             v_MWater[:,
                             d - 1] - v_MTranspiration - v_MWaterPhoto + v_MWaterUptake)  # nsamples by 1, (21)

        # Transpiration
        v_TempTranspiration = 1 - mean(exp(min(32 - E_AirTemp[:, 24 * d:24 * d + 24], tensor(0))
                                           * sum(g_TempTranspiration * ts, 1).view(nsamples, 1)),
                                       1)  # nsamples by 1, (23)

        temp5 = sum(g_HumidityTranspiration * ts, 1).view(nsamples, 1)
        temp5_2 = exp(-E_Humidity[:, 24 * d:24 * d + 24] * temp5)
        v_HumidityTranspiration = mean(temp5_2, 1)  # nsamples by 1, (24)

        v_EvapTranspiration = 1 - mean(sum(g_EvapTranspiration0 * ts, 1).view(nsamples, 1).repeat(1, 24)
                                       * exp(
            -(E_Evap[:, 24 * d:24 * d + 24] + 2.52e-3) * sum(g_EvapTranspiration1 * ts, 1).view(nsamples, 1)),
                                       1)  # nsamples by 1, (25)

        v_WindTranspiration = 1 - mean(sum(g_WindTranspiration0 * ts, 1).view(nsamples, 1).repeat(1, 24)
                                       * exp(
            -(E_Wind[:, 24 * d:24 * d + 24]) * sum(g_WindTranspiration1 * ts, 1).view(nsamples, 1)),
                                       1)  # nsamples by 1, (26)

        v_MTranspiration = v_MWater[:, d].clone() * v_TempTranspiration * v_HumidityTranspiration * v_EvapTranspiration * v_WindTranspiration  # nsamples by 1, (27)

        # Biomass
        v_PhotoLightCap = sum(g_Photo_Light * ts, 1) * sum(E_Light[:, 24 * d:24 * d + 24], 1) / M_Density * (1 - v_LightStress)   # nsamples by 1, (29)

        v_MPhotoWaterCap = sum(g_Photo_Water * ts, 1) * min(v_MWater[:, d] - v_MTranspiration,
                                                            sum(g_PhotoWater_Transpiration * ts,
                                                                1) * v_MTranspiration)  # nsamples by 1, (30)

        v_MPhotoLeafCap = sum(g_Photo_Leaf * ts, 1) * (1 - v_AirStress) * v_MLeafWeight[:, d - 1].clone()  # nsamples by 1, (32)

        v_MPhloemCap = sum(g_Phloem_Leaf * ts, 1) * v_MLeafWeight[:, d - 1].clone() + sum(g_Phloem_Stem * ts, 1) * v_MStemWeight[:, d - 1].clone()  # nsamples by 1, (34)

        v_MPhoto = \
        min(torch.stack([v_PhotoLightCap, v_MPhotoWaterCap, v_MPhotoLeafCap, v_MPhloemCap - v_MBiomass[:, d - 1]], -1),
            1)[0]  # nsamples by 1, (36)

        v_MLeafGrowthCap = (1 - v_AirStress) * min(
            sum(g_dLeaf0 * ts, 1) + sum(g_dLeaf1 * ts, 1) * v_MLeafWeight[:, d - 1].clone() + v_MLeafMS,
            sum(g_Phloem_Leaf * ts, 1) * v_MLeafWeight[:, d - 1].clone())  # nsamples by 1, (38)

        v_MRootGrowthCap = (1 - v_RootStress) * (sum(g_dRoot0 * ts, 1) + sum(g_dRoot1 * ts, 1) * v_MRootWeight[:,
                                                                                                 d - 1].clone() + v_MRootMS)  # nsamples by 1, (39)

        v_MStemGrowthCap = (1 - v_AirStress) * min(
            sum(g_dStem0 * ts, 1) + sum(g_dStem1 * ts, 1) * v_MStemWeight[:, d - 1].clone() + v_MStemMS,
            sum(g_Phloem_Stem * ts, 1) * v_MStemWeight[:, d - 1].clone())  # nsamples by 1, (40)

        v_MGrainGrowthCap = (1 - v_AirStress) * (sum(g_dGrain0 * ts, 1) + sum(g_dGrain1 * ts, 1) * v_MGrainWeight[:,
                                                                                                   d - 1].clone() + v_MGrainMS)  # nsamples by 1, (41)

        v_TasselGrowthCap = (1 - v_AirStress) * (sum(g_dTassel0 * ts, 1) + sum(g_dTassel1 * ts, 1) * v_TasselWeight[:,
                                                                                                   d - 1].clone() + v_TasselMS)  # nsamples by 1, (41)

        v_MGrowthCap = v_MLeafGrowthCap + v_MRootGrowthCap + v_MStemGrowthCap + v_MGrainGrowthCap  # nsamples by 1, (42)

        v_MGrowth = min(v_MBiomass[:, d - 1] + v_MPhoto, v_MGrowthCap)  # nsamples by 1, (48)

        # v_MBiomass = v_MBiomass.clone()
        v_MBiomass[:, d] = v_MBiomass[:, d - 1].clone() + v_MPhoto - v_MGrowth  # nsamples by 1, (50)

        v_MWaterPhoto = v_MPhoto / sum(g_Photo_Water * ts, 1)  # nsamples by 1, (52)

        # Maintenance and Senescence

        v_MLeafMS = min(sum(g_LeafMS0 * ts, 1) + sum(g_LeafMS1 * ts, 1) * v_AirStress, tensor(1)) * v_MLeafWeight[:, d - 1].clone()  # nsamples by 1, (54)

        v_MRootMS = min(sum(g_RootMS0 * ts, 1) + sum(g_RootMS1 * ts, 1) * v_RootStress, tensor(1)) * v_MRootWeight[:, d - 1].clone()  # nsamples by 1, (55)

        v_MStemMS = min(sum(g_StemMS0 * ts, 1) + sum(g_StemMS1 * ts, 1) * v_AirStress, tensor(1)) * v_MStemWeight[:, d - 1].clone()  # nsamples by 1, (56)

        v_MGrainMS = min(sum(g_GrainMS0 * ts, 1) + sum(g_GrainMS1 * ts, 1) * v_AirStress, tensor(1)) * v_MGrainWeight[:, d - 1].clone()  # nsamples by 1, (57)

        v_TasselMS = min(sum(g_TasselMS0 * ts, 1) + sum(g_TasselMS1 * ts, 1) * v_AirStress, tensor(1)) * v_TasselWeight[:, d - 1].clone()  # nsamples by 1, (57)

        # Growth
        # v_MLeafWeight = v_MLeafWeight.clone()
        v_MLeafWeight[:, d] = v_MLeafWeight[:, d - 1].clone() + max(- v_MLeafMS + v_MGrowth * v_MLeafGrowthCap / max(v_MGrowthCap, tensor(1)), tensor(0))  # nsamples by 1, (62), max(nan, 0) = 0 in case max growth = 0

        # v_MRootWeight[:, d] = v_MRootWeight[:, d - 1] + max(- v_MRootMS + v_MGrowth * fmax(v_MRootGrowthCap / v_MGrowthCap, tensor(0)), tensor(0))  # nsamples by 1, (63), max(nan, 0) = 0 in case max growth = 0
        # v_MRootWeight = v_MRootWeight.clone()
        v_MRootWeight[:, d] = v_MRootWeight[:, d - 1].clone() + max(- v_MRootMS + v_MGrowth * v_MRootGrowthCap / max(v_MGrowthCap, tensor(1)), tensor(0))  # nsamples by 1, (63), max(nan, 0) = 0 in case max growth = 0

        # v_MRootLength[:, d] = fmax(v_MRootLength[:, d - 1],
        #                            v_MRootWeight[:, d] / sum(g_RootDensity * ts, 1))  # nsamples by 1, (64)
        v_MRootLength_t = v_MRootLength[:, d - 1].clone()
        v_MRootLength = v_MRootLength.clone()
        v_MRootLength[:, d] = max(v_MRootLength_t, v_MRootWeight[:, d].clone() / sum(g_RootDensity * ts, 1))  # nsamples by 1, (64)

        v_MRootLayerLength = v_MRootLayerLength.clone()
        v_MRootLength_t = v_MRootLength[:, d].clone()
        v_MRootLayerLength[:, d, 0] = min(v_MRootLength_t, tensor(4 / 40))  # convert inches to meters
        v_MRootLayerLength[:, d, 1] = min(v_MRootLength_t - v_MRootLayerLength[:, d, 0], tensor(8 / 40))
        v_MRootLayerLength[:, d, 2] = min(v_MRootLength_t - sum(v_MRootLayerLength[:, d, 0:2], 1), tensor(12 / 40))
        v_MRootLayerLength[:, d, 3] = v_MRootLength_t - sum(v_MRootLayerLength[:, d, 0:3], 1)  # nsamples by 1, (65)

        # v_MStemWeight[:, d] = v_MStemWeight[:, d - 1] + max(- v_MStemMS + v_MGrowth * fmax(v_MStemGrowthCap / v_MGrowthCap, tensor(0)), tensor(0))  # nsamples by 130, (66)
        # v_MStemWeight = v_MStemWeight.clone()
        v_MStemWeight[:, d] = v_MStemWeight[:, d - 1].clone() + max(- v_MStemMS + v_MGrowth * v_MStemGrowthCap / max(v_MGrowthCap, tensor(1)), tensor(0))  # nsamples by 130, (66)

        v_MStemHeight_t = v_MStemHeight[:, d - 1].clone()
        v_MStemHeight = v_MStemHeight.clone()
        v_MStemHeight[:, d] = max(v_MStemHeight_t, v_MStemWeight[:, d].clone() / sum(g_StemDensity * ts, 1))  # nsamples by 130, (67)

        # v_MGrainWeight = v_MGrainWeight.clone()
        v_MGrainWeight[:, d] = v_MGrainWeight[:, d - 1].clone() + max(- v_MGrainMS + v_MGrowth * v_MGrainGrowthCap / max(v_MGrowthCap, tensor(1)), tensor(0))  # nsamples by 130, (68)

        # v_TasselWeight = v_TasselWeight.clone()
        v_TasselWeight[:, d] = v_TasselWeight[:, d - 1].clone() + max(- v_TasselMS + v_MGrowth * v_TasselGrowthCap / max(v_MGrowthCap, tensor(1)), tensor(0))  # nsamples by 130, (66)

        v_TasselHeight_t = v_TasselHeight[:, d - 1].clone()
        v_TasselHeight = v_TasselHeight.clone()
        v_TasselHeight[:, d] = max(v_TasselHeight_t, v_TasselWeight[:, d].clone() / sum(g_TasselDensity * ts, 1))  # nsamples by 130, (67)

        # v_MShootWeight = v_MShootWeight.clone()
        v_MShootWeight[:, d] = v_MLeafWeight[:, d].clone() + v_MStemWeight[:, d].clone() + v_MGrainWeight[:, d].clone() + v_TasselWeight[:, d].clone()  # nsamples by 130, (69)

        # HI
        v_HIStressLeaf = max(tensor(0), 1 - sum(g_AirStressIndexLeaf * ts, 1) * v_AirStress * v_HumidityStress * v_LightStress)
        v_HIStressStem = max(tensor(0), 1 - sum(g_AirStressIndexStem * ts, 1) * v_AirStress * v_HumidityStress)
        v_HIStressTassel = max(tensor(0), 1 - sum(g_AirStressIndexTassel * ts, 1) * v_AirStress * v_HumidityStress * v_LightStress)
        v_HIStressRoot = max(tensor(0), 1 - sum(g_AirStressIndexRoot * ts, 1) * v_RootStress)

        v_HILeafMS = min(sum(g_HILeafMS0 * ts, 1) + sum(g_HILeafMS1 * ts, 1) * v_AirStress * v_HumidityStress * v_LightStress, tensor(1)) * v_HIMatterCapLeaf[:, d - 1].clone()  # nsamples by 1, (54)
        v_HIRootMS = min(sum(g_HIRootMS0 * ts, 1) + sum(g_HIRootMS1 * ts, 1) * v_RootStress, tensor(1)) * v_HIMatterCapRoot[:, d - 1].clone()  # nsamples by 1, (55)
        v_HIStemMS = min(sum(g_HIStemMS0 * ts, 1) + sum(g_HIStemMS1 * ts, 1) * v_AirStress * v_HumidityStress, tensor(1)) * v_HIMatterCapStem[:, d - 1].clone()  # nsamples by 1, (56)
        v_HITasselMS = min(sum(g_HITasselMS0 * ts, 1) + sum(g_HITasselMS1 * ts, 1) * v_AirStress * v_HumidityStress * v_LightStress, tensor(1)) * v_HIMatterCapTassel[:, d - 1].clone()  # nsamples by 1, (57)

        v_HIMatterCapLeaf[:, d] = v_HIMatterCapLeaf[:, d - 1].clone() + max(sum(g_HI_Leaf * ts, 1) * v_MLeafWeight[:, d].clone() * v_HIStressLeaf - v_HILeafMS, tensor(0))
        v_HIMatterCapStem[:, d] = v_HIMatterCapStem[:, d - 1].clone() + max(sum(g_HI_Stem * ts, 1) * v_MStemWeight[:, d].clone() * v_HIStressStem - v_HIStemMS, tensor(0))
        v_HIMatterCapTassel[:, d] = v_HIMatterCapTassel[:, d - 1].clone() + max(sum(g_HI_Tassel * ts, 1) * v_TasselWeight[:, d].clone() * v_HIStressTassel - v_HITasselMS, tensor(0))
        v_HIMatterCapRoot[:, d] = v_HIMatterCapRoot[:, d - 1].clone() + max(sum(g_HI_Root * ts, 1) * v_MRootWeight[:, d].clone() * v_HIStressRoot - v_HIRootMS, tensor(0))

        v_HIPotentialLeaf = max(tensor(0), (v_HIMatterCapLeaf[:, d].clone() - sum(g_HIReferenceThresholdLeaf * ts, 1)) / sum(g_HIReferenceThresholdLeaf * ts, 1))
        v_HIPotentialStem = max(tensor(0), (v_HIMatterCapStem[:, d].clone() - sum(g_HIReferenceThresholdStem * ts, 1)) / sum(g_HIReferenceThresholdStem * ts, 1))
        v_HIPotentialTassel = max(tensor(0), (v_HIMatterCapTassel[:, d].clone() - sum(g_HIReferenceThresholdTassel * ts, 1)) / sum(g_HIReferenceThresholdTassel * ts, 1))
        v_HIPotentialRoot = max(tensor(0), (v_HIMatterCapRoot[:, d].clone() - sum(g_HIReferenceThresholdRoot * ts, 1)) / sum(g_HIReferenceThresholdRoot * ts, 1))

        v_HIPotential = v_HIPotential.clone()
        v_HIPotential[:, d] = torch.minimum(
            torch.minimum(v_HIPotentialLeaf, v_HIPotentialStem),
            torch.minimum(v_HIPotentialTassel, v_HIPotentialRoot)
        )

        if maternal_effect is None:
            maternal_health = sum(g_MaternalHealth * ts, 1)
            maternal_repair = sum(g_MaternalRepair * ts, 1)
        else:
            maternal_health = maternal_effect[:, 0]
            maternal_repair = maternal_effect[:, 1]
        v_HIRepairRate = exp(-maternal_health - maternal_repair)

        v_HIRate = v_HIRate.clone()
        repair_weight = 0.5
        v_HIRate[:, d] = torch.tanh(g_start0 + g_start1 * v_HIPotential[:, d].clone()) * (1 - repair_weight * v_HIRepairRate)

    index = tensor([i * 130 + M_Harvest.long()[i] - 1 for i in range(nsamples)]).to(device)

    HarvestHeight = v_MStemHeight.take(index)  # in meter

    ph["Height"] = v_MStemHeight  # in meter
    ph["Height"] = ph["Height"] + (ph["Height"] == 0) * HarvestHeight.view(nsamples, 1)
    ph['HIR'] = v_HIRate
    ph['TasselLength'] = v_TasselHeight
    ph['stage'] = ts_all


    # for key in ph.keys():
    #     ph[key] = ph[key].to('cpu')

    return ph


#%%
if __name__ == "__main__":
    #%%
    # torch.autograd.set_detect_anomaly(True)
    # torch.autograd.set_detect_anomaly(False)

    # 设置设备为 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    # 参数
    nsamples = 1200
    H = 4320
    genostage = 3
    genoparam = 76
    envivar = 13

    # 构造输入数据
    envi = (torch.randn(nsamples, H, envivar) * 10 + 50).to(device)  # 环境变量
    mg = torch.cat([
        torch.randint(low=160, high=170, size=(nsamples, 1)),          # 收获日 160 ~ 130
        torch.rand(nsamples, 1) * 3 + 3,                               # 密度 3.0 ~ 6.0
        torch.randint(low=0, high=5, size=(nsamples, 1))
    ], dim=1).to(device)
    geno = torch.rand(nsamples, genostage, genoparam).to(device)       # 遗传参数

    def generate_ideal_height_curve(nsamples=2000, device='cpu'):
        days = torch.arange(130, device=device).float()  # 0~179天
        max_height = torch.normal(mean=2.0, std=0.2, size=(nsamples, 1)).to(device)  # 每株植物最终高度在 1.8~2.2 米之间

        # 生长曲线函数：sigmoid 形式
        growth_rate = 0.1
        turning_point = 90.0  # 拐点在90天左右

        height_curve = max_height * torch.sigmoid(growth_rate * (days - turning_point))  # [nsamples, 130]
        return height_curve

    target_height = generate_ideal_height_curve(nsamples=nsamples, device='cuda')
    target_HIR = torch.rand(nsamples, 1, device='cuda') / 4
    geno = torch.rand(nsamples, 3, genoparam, device=device, requires_grad=True)


    #%%
    # 运行 dhcorn 模型
    ph = dhcorn(envi, mg, geno, device=device)


    #%%
    import gc
    # 可训练参数 geno 初始化
    # geno = geno.repeat(nsamples, 1, 1)

    optimizer = torch.optim.Adam([geno], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()

        ph = dhcorn(envi, mg, geno, device=device)
        loss = torch.nn.functional.mse_loss(ph["Height"], target_height) / target_height.mean()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1:02d}, HIR RRMSE Loss = {loss.item():.4f}")

        torch.cuda.empty_cache()
        gc.collect()

