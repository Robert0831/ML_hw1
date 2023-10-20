import pandas as pd
import numpy as np
class dataset():
    def __init__(self,file_path):
        self.data=pd.read_csv(file_path)
        self.header = list(self.data.columns)

        self.data['area_cluster'] = self.data['area_cluster'].replace({'C1': 0, 'C2': 1,'C3': 2,'C4': 3,'C5': 4,'C6': 5,'C7': 6,'C8': 7,'C9': 8,'C10': 9,'C11': 10,
        'C12': 11, 'C13': 12,'C14': 13,'C15': 14,'C16': 15,'C17': 16,'C18': 17,'C19': 18,'C20': 19,'C21': 20,'C22': 21,})
        self.data['segment'] = self.data['segment'].replace({ 'Utility':0,'A':1,'B1':2,'B2':3,'C1':4,'C2':5})
        self.data['model'] = self.data['model'].replace({'M1':0,'M2':1,'M3':2,'M4':3,'M5':4,'M6':5,'M7':6,'M8':7,'M9':8,'M10':9,'M11':10})
        self.data['fuel_type'] = self.data['fuel_type'].replace({'CNG':0,'Petrol':1,'Diesel':2})
        self.data['max_torque'] = self.data['max_torque'].replace({'60Nm@3500rpm':0,'113Nm@4400rpm':1,'170Nm@4000rpm':2,'200Nm@1750rpm':3,'200Nm@3000rpm':4,'250Nm@2750rpm':5,'82.1Nm@3400rpm':6,'85Nm@3000rpm':7,'91Nm@4250rpm':8})
        self.data['max_power'] = self.data['max_power'].replace({'40.36bhp@6000rpm':0,'118.36bhp@5500rpm':1,'113.45bhp@4000rpm':2,'55.92bhp@5300rpm':3,'61.68bhp@6000rpm':4,'67.06bhp@5500rpm':5,'88.50bhp@6000rpm':6,'88.77bhp@4000rpm':7,'97.89bhp@3600rpm':8})
        self.data['engine_type'] = self.data['engine_type'].replace({'1.0 SCe':0,'1.2 L K Series Engine':1,'1.2 L K12N Dualjet':2,'1.5 L U2 CRDi':3,'1.5 Turbocharged Revotorq':4,'1.5 Turbocharged Revotron':5,'F8D Petrol Engine':6,'G12B':7,'i-DTEC':8,'K Series Dual jet':9,'K10C':10})
        self.data['is_esc'] = self.data['is_esc'].replace({'Yes': 1, 'No': 0})
        self.data['is_adjustable_steering'] = self.data['is_adjustable_steering'].replace({'Yes': 1, 'No': 0})
        self.data['is_tpms'] = self.data['is_tpms'].replace({'Yes': 1, 'No': 0})
        self.data['is_parking_sensors'] = self.data['is_parking_sensors'].replace({'Yes': 1, 'No': 0})
        self.data['is_parking_camera'] = self.data['is_parking_camera'].replace({'Yes': 1, 'No': 0})
        self.data['rear_brakes_type'] = self.data['rear_brakes_type'].replace({'Drum': 1, 'Disc': 0})
        self.data['transmission_type'] = self.data['transmission_type'].replace({'Automatic': 1, 'Manual': 0})
        self.data['steering_type'] = self.data['steering_type'].replace({'Electric': 0, 'Power': 1,'Manual': 2})
        self.data['is_front_fog_lights'] = self.data['is_front_fog_lights'].replace({'Yes': 1, 'No': 0})
        self.data['is_rear_window_wiper'] = self.data['is_rear_window_wiper'].replace({'Yes': 1, 'No': 0})
        self.data['is_rear_window_washer'] = self.data['is_rear_window_washer'].replace({'Yes': 1, 'No': 0})
        self.data['is_rear_window_defogger'] = self.data['is_rear_window_defogger'].replace({'Yes': 1, 'No': 0})
        self.data['is_brake_assist'] = self.data['is_brake_assist'].replace({'Yes': 1, 'No': 0})
        self.data['is_power_door_locks'] = self.data['is_power_door_locks'].replace({'Yes': 1, 'No': 0})
        self.data['is_central_locking'] = self.data['is_central_locking'].replace({'Yes': 1, 'No': 0})

        self.data['is_power_steering'] = self.data['is_power_steering'].replace({'Yes': 1, 'No': 0})
        self.data['is_driver_seat_height_adjustable'] = self.data['is_driver_seat_height_adjustable'].replace({'Yes': 1, 'No': 0})
        self.data['is_day_night_rear_view_mirror'] = self.data['is_day_night_rear_view_mirror'].replace({'Yes': 1, 'No': 0})
        self.data['is_ecw'] = self.data['is_ecw'].replace({'Yes': 1, 'No': 0})
        self.data['is_speed_alert'] = self.data['is_speed_alert'].replace({'Yes': 1, 'No': 0})


        self.data=self.data.sample(frac=1, random_state=101)  # frac=1 means shuffling the entire DataFrame
        self.y=self.data['is_claim']
        self.data.drop(columns=['policy_id','is_claim'], inplace=True)
        self.row=self.data.shape[0] #58592

        train_index=int(self.row*0.8)
        val_index=int(self.row*0.9)
        
        self.train_data=self.data.iloc[0:int(train_index*0.02)].to_numpy()
        max=np.max(self.train_data,axis=0)
        min=np.min(self.train_data,axis=0)
        self.train_data=(self.train_data-min)/(max-min)

        self.val_data=self.data.iloc[train_index:int(train_index+val_index*0.02)].to_numpy()
        self.val_data=(self.val_data-min)/(max-min)

        self.test_data=self.data.iloc[val_index:].to_numpy()
        self.test_data=(self.test_data-min)/(max-min)


        self.train_label=self.y.iloc[0:int(train_index*0.02)].to_numpy()
        self.val_label=self.y.iloc[train_index:int(train_index+val_index*0.02)].to_numpy()
        self.test_label=self.y.iloc[val_index:].to_numpy()



if __name__ == '__main__':
    file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"
    a=dataset(file_path)


    # for index, (rows,labels) in enumerate(zip(a.train_data.iterrows(),a.train_label)):
    #     _, row = rows
    #     row=row.values
    #     print(row)
    #     print(labels)
    #     break

    for index, (row,labels) in enumerate(zip(a.train_data,a.train_label)):

        print(row)
        print(labels)
        aa=np.random.random(42)
        print(np.dot(aa,row))

        break