import h5py
import matplotlib.pyplot as plt
import csv

# 打开HDF5文件
with h5py.File('data/thor_flight121_studentgnssdeniednavfilter_2014_05_10.h5', 'r') as file:
    # ['GPS_TOW', 'Pd', 'Pd_aoa', 'Pd_aos', 'Ps', 'adStatus', 'alt', 'aoa', 'aos', 'ax', 'ax_bias', 'ay', 'ay_bias',
    # 'az', 'az_bias', 'cpuLoad', 'da_l', 'da_l_in', 'da_r', 'da_r_in', 'de', 'de_in', 'df_l', 'df_l_in', 'df_r',
    # 'df_r_in', 'dr', 'dr_in', 'dthr', 'dthr_in', 'etime_actuators', 'etime_control', 'etime_daq', 'etime_datalog',
    # 'etime_guidance', 'etime_nav', 'etime_sensfault', 'etime_surffault', 'etime_sysid', 'etime_telemetry',
    # 'gamma_cmd', 'gndtrk_cmd', 'gpsStatus', 'h', 'h_cmd', 'h_filt', 'hx', 'hy', 'hz', 'ias', 'ias_cmd', 'ias_filt',
    # 'imuStatus', 'lat', 'lon', 'mode', 'navValid', 'navalt', 'navlat', 'navlon', 'navvd', 'navve', 'navvn', 'p',
    # 'p_bias', 'p_cmd', 'phi', 'phi_cmd', 'psi', 'psi_cmd', 'q', 'q_bias', 'q_cmd', 'r', 'r_bias', 'r_cmd',
    # 'run_num', 'satVisible', 'theta', 'theta_cmd', 'time', 'vd', 've', 'vn']

    if 'alt' in file:
        alt_data = file['alt'][:]
        print("高度数据:", alt_data)
        print("Shape of 'alt' dataset:", alt_data.shape)

        # 将数据保存为CSV文件
        csv_filename = 'flight121.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # 写入表头
            headers = ['GPS_TOW', 'time', '高度', '攻角', '法向加速度', '空速', '滚转角', '俯仰角']
            csv_writer.writerow(headers)

            # 写入数据
            GPS_TOW_value = file['GPS_TOW'][:][0]
            time_values = file['time'][:][0]
            alt_values = alt_data[0]
            aoa_values = file['aoa'][:][0]
            az_bias_values = file['az_bias'][:][0]
            ias_values = file['ias'][:][0]
            phi_values = file['phi'][:][0]
            theta_values = file['theta'][:][0]

            data_rows = zip(GPS_TOW_value, time_values, alt_values, aoa_values, az_bias_values, ias_values, phi_values,
                            theta_values)
            csv_writer.writerows(data_rows)

        print(f"数据已保存为'{csv_filename}'文件。")

    else:
        print("未找到数据集 'alt'。")

# plt.plot(range(len(alt_data[0][3800:13800])), alt_data[0][3800:13800])
# # plt.title('高度随时间变化')
# plt.xlabel('time')
# plt.ylabel('H')
# plt.ylim(0, 420)  # 设置y轴范围
# plt.show()
# plt.pause(0)
