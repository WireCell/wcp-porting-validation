## apa_cpu: google-pprof --text (flat, top 30)

```
Total: 16991 samples
    5817  34.2%  34.2%     7403  43.6% WireCell::Img::BlobDepoFill::operator
     820   4.8%  39.1%     2032  12.0% std::_Rb_tree::_M_insert_unique
     742   4.4%  43.4%      742   4.4% std::_Rb_tree_increment@@GLIBCXX_3.4@bfcb0
     487   2.9%  46.3%      722   4.2% std::_Rb_tree::_S_right (inline)
     480   2.8%  49.1%      678   4.0% WireCell::LassoModel::Fit
     385   2.3%  51.4%      385   2.3% tc_new
     354   2.1%  53.5%      354   2.1% std::_Rb_tree_insert_and_rebalance@@GLIBCXX_3.4
     303   1.8%  55.3%      304   1.8% tcmalloc::CentralFreeList::FetchFromOneSpans
     294   1.7%  57.0%      698   4.1% std::__new_allocator::deallocate (inline)
     259   1.5%  58.5%      888   5.2% std::_Rb_tree::_M_erase (inline)
     211   1.2%  59.7%     1091   6.4% std::__cxx11::basic_string::basic_string (inline)
     207   1.2%  61.0%      207   1.2% 0x00007fffe06fba29
     194   1.1%  62.1%      194   1.1% tc_delete_sized
     185   1.1%  63.2%     2174  12.8% std::_Destroy_aux::__destroy (inline)
     170   1.0%  64.2%      972   5.7% std::_Rb_tree::_M_erase
     167   1.0%  65.2%      187   1.1% tcmalloc::CentralFreeList::Populate
     162   1.0%  66.1%      192   1.1% std::__detail::_Mod_range_hashing::operator (inline)
     139   0.8%  67.0%      143   0.8% BZ2_decompress
     137   0.8%  67.8%      403   2.4% std::unordered_map::operator[] (inline)
     137   0.8%  68.6%      149   0.9% tcmalloc::CentralFreeList::ReleaseToSpans
     130   0.8%  69.3%      130   0.8% WireCell::Aux::SimpleBlob::shape
     122   0.7%  70.0%      122   0.7% WireCell::Aux::BlobShadow::shadow_list (inline)
     106   0.6%  70.7%      106   0.6% unRLE_obuf_to_output_FAST (inline)
      97   0.6%  71.2%       97   0.6% __memcmp_evex_movbe
      97   0.6%  71.8%      654   3.8% std::_Rb_tree::_M_get_insert_unique_pos (inline)
      94   0.6%  72.4%      267   1.6% std::_Rb_tree::_M_copy
      83   0.5%  72.9%      623   3.7% std::__new_allocator::allocate (inline)
      79   0.5%  73.3%       79   0.5% __memcpy_avx512_unaligned_erms
      78   0.5%  73.8%      263   1.5% std::vector::operator[] (inline)
      75   0.4%  74.2%       91   0.5% std::__detail::_Compiler::_M_match_token
```

## apa_cpu: google-pprof --text --cum (WireCell/Eigen frames, top 40)

```
Total: 16991 samples
       0   0.0%   0.0%    16734  98.5% WireCell::Main::operator
       3   0.0%   0.0%    16734  98.5% WireCell::Pgraph::Graph::execute
       0   0.0%   0.0%    16734  98.5% WireCell::Pgraph::Pgrapher::execute
       0   0.0%   0.0%    16483  97.0% WireCell::Pgraph::Graph::call_node
       0   0.0%   0.0%     7411  43.6% WireCell::Pgraph::JoinFanin::operator
       0   0.0%   0.0%     7403  43.6% WireCell::IJoinNode::operator
    5817  34.2%  34.3%     7403  43.6% WireCell::Img::BlobDepoFill::operator
       0   0.0%  34.3%     6722  39.6% WireCell::Pgraph::Function::operator
       0   0.0%  34.3%     6028  35.5% WireCell::IFunctionNode::operator
       1   0.0%  40.6%     1497   8.8% WireCell::Pgraph::Sink::operator
       0   0.0%  40.6%     1389   8.2% WireCell::Sio::ClusterFileSink::operator
       0   0.0%  40.6%     1363   8.0% WireCell::Sio::ClusterFileSink::numpify
       1   0.0%  40.6%     1293   7.6% WireCell::cluster_node_t::cluster_node_t (inline)
       1   0.0%  40.6%     1290   7.6% WireCell::Img::ProjectionDeghosting::operator
       2   0.0%  40.6%     1268   7.5% WireCell::Aux::ClusterArrays::to_arrays
       0   0.0%  40.7%     1110   6.5% WireCell::Img::ChargeSolving::configure
       0   0.0%  45.9%      800   4.7% WireCell::Aux::SimpleCluster::~SimpleCluster (inline)
       0   0.0%  45.9%      772   4.5% WireCell::RayGrid::associate
       0   0.0%  50.2%      724   4.3% WireCell::Img::CS::solve
       0   0.0%  53.2%      715   4.2% WireCell::Img::InSliceDeghosting::operator
       5   0.0%  55.0%      685   4.0% WireCell::Aux::ClusterArrays::bodge_channel_slice
       0   0.0%  55.0%      679   4.0% WireCell::IndexedSet::size (inline)
       0   0.0%  55.0%      679   4.0% WireCell::Ress::solve
     480   2.8%  57.8%      678   4.0% WireCell::LassoModel::Fit
       5   0.0%  58.5%      647   3.8% WireCell::NamedFactory::find
       0   0.0%  58.5%      639   3.8% WireCell::Img::LocalGeomClustering::operator
       3   0.0%  58.5%      633   3.7% WireCell::RayGrid::overlap [clone .localalias]
      10   0.1%  59.1%      620   3.6% WireCell::Aux::BlobShadow::shadow_list
       0   0.0%  59.1%      573   3.4% WireCell::IQueuedoutNode::operator
       0   0.0%  59.1%      573   3.4% WireCell::Pgraph::Queuedout::operator
       2   0.0%  59.1%      559   3.3% WireCell::Img::ChargeSolving::operator
       1   0.0%  59.2%      526   3.1% WireCell::Img::CS::repack
       1   0.0%  59.4%      494   2.9% WireCell::Img::grouped_geom_clustering
       0   0.0%  59.8%      459   2.7% WireCell::Img::BlobClustering::flush
       0   0.0%  59.8%      459   2.7% WireCell::Img::BlobClustering::operator
       4   0.0%  59.8%      436   2.6% WireCell::Img::Projection2D::get_projection
      39   0.2%  60.5%      426   2.5% WireCell::RayGrid::projection
      43   0.3%  60.7%      423   2.5% WireCell::Img::BlobDepoFill::operator (inline)
      17   0.1%  66.0%      343   2.0% WireCell::Aux::dumps[abi:cxx11]
       7   0.0%  66.2%      333   2.0% WireCell::Img::Projection2D::dump[abi:cxx11]
```

## sub4f_cpu: google-pprof --text (flat, top 30)

```
Total: 295399 samples
   71637  24.3%  24.3%    80462  27.2% WireCell::Img::BlobDepoFill::operator
   36538  12.4%  36.6%   117595  39.8% WireCell::LassoModel::Fit
   18629   6.3%  42.9%    19719   6.7% Eigen::SparseMatrix::operator=
   15223   5.2%  48.1%    26473   9.0% std::vector::emplace_back (inline)
   11190   3.8%  51.9%    39434  13.3% Eigen::internal::set_from_triplets (inline)
    8439   2.9%  54.7%     8439   2.9% std::_Rb_tree_increment@@GLIBCXX_3.4@bfcb0
    7510   2.5%  57.3%    15752   5.3% std::_Rb_tree::_M_insert_unique
    7438   2.5%  59.8%     7449   2.5% WireCell::Aux::BlobShadow::shadow_list (inline)
    6321   2.1%  61.9%     6481   2.2% std::__detail::_Mod_range_hashing::operator (inline)
    6214   2.1%  64.0%    11484   3.9% std::__new_allocator::deallocate (inline)
    5687   1.9%  66.0%     5708   1.9% Eigen::SparseMatrix::insertBackUncompressed (inline)
    4780   1.6%  67.6%     5344   1.8% std::__unguarded_partition (inline)
    4647   1.6%  69.1%     5199   1.8% std::__relocate_a_1 (inline)
    4615   1.6%  70.7%     5863   2.0% _M_rehash_aux (inline)
    4095   1.4%  72.1%     4095   1.4% tc_new
    3246   1.1%  73.2%     3246   1.1% std::_Rb_tree_insert_and_rebalance@@GLIBCXX_3.4
    3117   1.1%  74.2%     4744   1.6% std::_Rb_tree::_S_right (inline)
    2957   1.0%  75.2%    16563   5.6% _M_find_before_node (inline)
    2874   1.0%  76.2%     2874   1.0% _S_equals (inline)
    2598   0.9%  77.1%     6443   2.2% Eigen::internal::redux_impl::run (inline)
    2415   0.8%  77.9%     2415   0.8% tc_delete_sized
    2157   0.7%  78.6%     2251   0.8% tcmalloc::CentralFreeList::ReleaseToSpans
    2027   0.7%  79.3%     2042   0.7% Eigen::MatrixBase::dot (inline)
    2016   0.7%  80.0%     3633   1.2% std::__adjust_heap
    1787   0.6%  80.6%     8731   3.0% WireCell::RayGrid::projection
    1693   0.6%  81.2%     1716   0.6% _mm_mul_pd (inline)
    1678   0.6%  81.8%     2032   0.7% tcmalloc::CentralFreeList::Populate
    1616   0.5%  82.3%     2556   0.9% Eigen::SparseMatrix::collapseDuplicates
    1603   0.5%  82.9%     4384   1.5% std::unordered_map::operator[] (inline)
    1570   0.5%  83.4%     1570   0.5% WireCell::Aux::SimpleBlob::shape@d8ae0
```

## sub4f_cpu: google-pprof --text --cum (WireCell/Eigen frames, top 40)

```
Total: 295399 samples
       0   0.0%   0.0%   295099  99.9% WireCell::Main::operator
       3   0.0%   0.0%   295099  99.9% WireCell::Pgraph::Graph::execute
       0   0.0%   0.0%   295099  99.9% WireCell::Pgraph::Pgrapher::execute
       0   0.0%   0.0%   294841  99.8% WireCell::Pgraph::Graph::call_node
       0   0.0%   0.0%   201713  68.3% WireCell::Pgraph::Function::operator
       0   0.0%   0.0%   196571  66.5% WireCell::IFunctionNode::operator
       2   0.0%   0.0%   123176  41.7% WireCell::Img::ChargeSolving::configure
      83   0.0%   0.2%   118288  40.0% WireCell::Img::CS::solve
       0   0.0%   0.2%   117604  39.8% WireCell::IndexedSet::size (inline)
       0   0.0%   0.2%   117603  39.8% WireCell::Ress::solve
   36538  12.4%  12.6%   117595  39.8% WireCell::LassoModel::Fit
       0   0.0%  12.6%    80469  27.2% WireCell::Pgraph::JoinFanin::operator
       0   0.0%  12.6%    80462  27.2% WireCell::IJoinNode::operator
   71637  24.3%  36.8%    80462  27.2% WireCell::Img::BlobDepoFill::operator
       0   0.0%  36.9%    39434  13.3% Eigen::SparseMatrix::setFromTriplets (inline)
   11190   3.8%  40.7%    39434  13.3% Eigen::internal::set_from_triplets (inline)
      13   0.0%  41.2%    36602  12.4% WireCell::cluster_node_t::cluster_node_t (inline)
     148   0.1%  41.2%    34218  11.6% WireCell::Aux::BlobShadow::shadow_list
   18629   6.3%  52.7%    19719   6.7% Eigen::SparseMatrix::operator=
       0   0.0%  52.7%    17859   6.0% WireCell::RayGrid::associate
      48   0.0%  56.7%    15102   5.1% WireCell::NamedFactory::find
      42   0.0%  56.7%    14977   5.1% WireCell::RayGrid::overlap [clone .localalias]
      20   0.0%  56.8%    12908   4.4% WireCell::Img::grouped_geom_clustering
       1   0.0%  59.3%    10261   3.5% WireCell::Img::LocalGeomClustering::operator
       2   0.0%  59.4%     8838   3.0% WireCell::Img::InSliceDeghosting::operator
    1787   0.6%  60.0%     8731   3.0% WireCell::RayGrid::projection
    7438   2.5%  65.4%     7449   2.5% WireCell::Aux::BlobShadow::shadow_list (inline)
     323   0.1%  65.6%     6766   2.3% Eigen::DenseBase::sum (inline)
      45   0.0%  65.6%     6690   2.3% Eigen::MatrixBase::dot
       0   0.0%  65.6%     6656   2.3% Eigen::internal::dot_nocheck::run (inline)
       0   0.0%  65.6%     6559   2.2% WireCell::IQueuedoutNode::operator
       0   0.0%  65.6%     6559   2.2% WireCell::Pgraph::Queuedout::operator
       0   0.0%  67.7%     6446   2.2% WireCell::Img::BlobClustering::operator
       0   0.0%  67.7%     6444   2.2% WireCell::Img::BlobClustering::flush
       0   0.0%  67.7%     6443   2.2% Eigen::DenseBase::redux (inline)
    2598   0.9%  68.6%     6443   2.2% Eigen::internal::redux_impl::run (inline)
       0   0.0%  70.6%     5819   2.0% WireCell::Pgraph::Sink::operator
       0   0.0%  71.0%     5709   1.9% WireCell::Img::geom_clustering
    5687   1.9%  73.0%     5708   1.9% Eigen::SparseMatrix::insertBackUncompressed (inline)
       0   0.0%  73.0%     5654   1.9% WireCell::Aux::SimpleCluster::~SimpleCluster (inline)
```

