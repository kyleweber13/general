from nwposture_dev.nwposture.NWPosture3 import NWPosture, detect_sts, fill_between_walks, calculate_bouts
from nwposture_dev.nwposture.NWPosture3 import apply_logic, fill_other, final_logic, plot_posture_comparison
import numpy as np
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")

# Import data in nwposture_development2.py


""" ===================================== ANGLE AND BASIC POSTURE DETECTION ======================================= """
post = NWPosture(chest_dict=chest, ankle_dict=ankle, chest_acc_units='g', study_code='OND09', subject_id='Test', coll_id='01')
# post = NWPosture(chest_dict=chest, ankle_dict=ankle, study_code='OND09', subject_id=subj, coll_id='01')

post.crop_data()
post.df_gait = post.load_gait_data(df_gait)
post.df_gait = post.df_gait.loc[post.df_gait['start_timestamp'] > df_event['start_timestamp'].iloc[0]]
post.gait_mask = post.create_gait_mask()
df1s = post.calculate_postures(goldstandard_df=df_event)

first_walk_index = df1s.loc[df1s['chest_gait_mask'] == 1].iloc[0]['index']


""" ======================================= SIT/STAND TRANSITION DETECTION ======================================== """


df_sts = detect_sts(nwposture_obj=post, remove_lessthan=2, pad_len=1)


""" ===================================== FILLING SMALL GAPS BETWEEN WALKS ======================================== """

df1s['v1'] = fill_between_walks(postures=df1s['posture'],  gait_mask=df1s['chest_gait_mask'], df_transitions=df_sts,
                                max_break=5, pad_walks=1)


""" ================================================ APPLYING LOGIC =============================================== """

# First datapoint for each posture
df_bout = calculate_bouts(df1s=df1s, colname='v1')

df1s['v2'], df_logic1 = apply_logic(df_transitions=df_sts, gait_mask=df1s['chest_gait_mask'],
                                    first_walk_index=first_walk_index, postures=df1s['v1'],
                                    first_pass=True, quiet=False)

df1s['v3'], df_logic2 = apply_logic(df_transitions=df_sts, gait_mask=df1s['chest_gait_mask'],
                                    first_walk_index=first_walk_index, postures=df1s['v2'],
                                    first_pass=False, quiet=False)

df1s['v4'] = final_logic(postures=df1s['v3'], df_transitions=df_sts, gait_mask=df1s['chest_gait_mask'])
df1s['v4'] = fill_other(postures=df1s['v4'])

""" =============================================== PLOTTING RESULTS ============================================== """

# fig = plot_posture_comparison(df1s=df1s, show_transitions=False, show_v0=False, show_v2=False, show_v3=False, show_v4=True, show_gs=True, use_timestamps=False, collapse_lying=False)


def plot_for_communityengagement_day():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(sharex='col', figsize=(10, 8))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 4)
    ax3 = plt.subplot(2, 3, 2)
    ax4 = plt.subplot(2, 3, 5)
    ax5 = plt.subplot(1, 3, 3)

    ax1.plot(np.arange(len(chest['Up'][:200000]))/100, chest['Up'][:200000], color='black', label='Trunk (vertical)')
    # ax1.legend()
    ax1.grid()
    # ax1.set_xticklabels([])
    ax1.set_title("Trunk - Vertical Acceleration")
    ax1.set_xlim(400, 1100)
    ax1.set_ylim(-1, 3)

    ax2.plot(np.arange(len(ankle['Up'][:200000]))/100, ankle['Up'][:200000], color='red', label='Shin (vertical)')
    # ax2.legend()
    ax2.grid()
    ax2.set_title("Shin - Vertical Acceleration")
    ax2.set_xlabel("Seconds")
    ax2.set_xlim(400, 1100)
    ax2.set_ylim(-1, 3)

    ax3.plot(np.arange(2000), df1s['chest_up'].iloc[:2000], color='black', label='Trunk (vertical)')
    # ax4.legend()
    ax3.grid()
    ax3.set_title("Trunk - Vertical Angle")
    # ax3.set_xticklabels([])
    ax3.set_xlim(400, 1100)

    ax4.plot(np.arange(2000), df1s['ankle_up'].iloc[:2000], color='red', label='Shin (vertical)')
    # ax4.legend()
    ax4.grid()
    ax4.set_title("Shin - Vertical Angle")
    ax4.set_xlabel("Seconds")
    ax4.set_xlim(400, 1100)

    ax5.plot(np.arange(2000), df1s['GS'].iloc[:2000], color='green', label='Posture')
    ax5.grid()
    ax5.set_xlabel("Seconds")
    ax5.set_xlim(400, 1100)
    ax5.set_title("Final Posture")

    plt.tight_layout()