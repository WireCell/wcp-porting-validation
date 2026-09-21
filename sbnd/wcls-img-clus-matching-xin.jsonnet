// SBND LArSoft 1-step chain (imaging + clustering + Q/L + the PR tail) -- THIN RE-EXPORT.
//
// The implementation now lives in the canonical, in-tree job
//   cfg/pgrapher/experiment/sbnd/wcls-img-clus-matching-xin.jsonnet
// promoted out of the working repo by doc sbnd_xin/120 sec 4 on the owner's ask
// 2026-09-21, together with the deletion of pr-operating-point.jsonnet: the SBND
// production operating point is now pr()'s own defaults, so this chain no longer
// needs anything from the working repo to reconstruct at the production point.
//
// This file stays as a thin re-export so every fcl that names it by path, and
// any setup script that puts wcp-porting-img/sbnd on WIRECELL_PATH, keeps
// working unchanged.  extVars bind through the import untouched.
//
// EDIT THE IN-TREE FILE, NOT THIS ONE.

import 'pgrapher/experiment/sbnd/wcls-img-clus-matching-xin.jsonnet'
