


state_anomaly = ["damaged {}",
                 "flawed {}",
                 "abnormal {}",
                 "imperfect {}",
                 "blemished {}",
                 "{} with flaw",
                 "{} with defect",
                 "{} with damage"]

abnormal_state0 = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

#
class_state_abnormal = {

    'bagel': ['{} with defect', '{} with contamination', '{} with crack', '{} with hole'],
    'cable_gland': ['{} with a bent shape', '{} with cut', '{} with hole', '{} with thread residue'],
    'carrot': ['{} with defect', '{} with contamination', '{} with crack', '{} with cut', '{} with hole'], 
    'cookie': ['{} with defect', '{} with contamination', '{} with crack', '{} with hole'], 
    'dowel': ['{} with a bent shape', '{} with defect', '{} with contamination', '{} with cut'],
    'foam': ['{} with color spot', '{} with defect', '{} with contamination', '{} with cut'], 
    'peach': ['{} with defect', '{} with contamination', '{} with cut', '{} with hole'], 
    'potato': ['{} with defect', '{} with contamination', '{} with cut', '{} with hole'], 
    'rope': ['{} with contamination', '{} with cut', '{} with open part'], 
    'tire': ['{} with defect', '{} with contamination', '{} with cut', '{} with hole']
}
