function TensorDecomposition(cp_rank)
  addpath('./tensor_toolbox');

  tensor_info = load('tmp_tensor_info.mat');

  tensor = sptensor(double(tensor_info.coord_list)+1, double(tensor_info.val_list'), [double(tensor_info.vocab_size), double(tensor_info.vocab_size), ...
  double(tensor_info.doc_size)]);

  % cp is better
  %factors = tucker_als(tensor, double(cp_rank));
  factors = cp_als(tensor, double(cp_rank));

  doc2vec = factors.U{3};

  save('tmp_doc2vec_mat.mat', 'doc2vec');
end