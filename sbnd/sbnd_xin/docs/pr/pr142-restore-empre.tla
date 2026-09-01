# doc pr/142 -- restore the EM/pi0 campaign knobs (docs pr/117-141) to their
# pre-campaign values.  Derived, not hand-listed: the TLA defaults of
# cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet at 8d93260d (2026-08-25,
# the last commit before pr/117) vs ddce7430 (HEAD), minus the 12 doc-84
# MCS / long-muon knobs, which the owner scoped to stay ON in BOTH arms.
# Suppressing value taken from each key's emission guard, so the compiled
# JSON drops the key exactly as the pre-campaign config did.
# Consumed by run_pr_chain_batch.sh's PR_EXTRA_TLA (line 1800).
# 39 entries.  (kine_guard_freed_miss_deg needs no entry: its key is emitted only
#  when kine_guard_freed_impact != 0, which this file already sets to 0.)
kine_count_conn4_near=false
kine_count_guard_freed=false
kine_count_near_cross_cluster=false
kine_guard_freed_impact=0
kine_shower_fudge_factor=null
pf_conn4_near_candidate=false
pf_orphan_guard_freed=false
pf_orphan_near_cross_cluster=false
pi0_admit_muon_showers=false
pi0_admit_type3=false
pi0_attached_partner_min_mev=0
pi0_bp_vertex_miss_cm=null
pi0_nc_floor_mev=null
pi0_nc_frag_merge=false
pi0_nc_pf_assoc_deg=null
pi0_nc_sig_angle_deg=null
pi0_prefer_main_vertex=false
pi0_readmit_retyped=false
sccc_max_gap=6
shower_em_collinear_deg=null
shower_em_collinear_dis_cm=null
shower_ex1_dedup_rehome=false
shower_merge_relax=false
shower_merge_relax_continuity=false
shower_pass3_backfill_guard_len=0
shower_pass3_cone_guard_len=0
shower_pass4_best_owner=false
shower_pass4_prefilter_v1_escape=false
shower_pass4_prefilter_v1_max_v2=0
shower_pass4_prox_guard_len=0
shower_pass4_prune_detached=false
shower_pass4_prune_gap2=0
shower_pass4_track_guard_len=0
shower_samevtx_track_absorb=false
shower_satellite_absorb=false
shower_split=false
shower_split_em_start=false
stem_backfill_back_dvtx=0
stem_backfill_back_guard=false
