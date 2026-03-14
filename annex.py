import hnswlib
import time
import numpy as np
import threading
import pickle
import faiss
from .filter_functions.filter_map import filter_map
from typing import List


class AnnexVec():
    def __init__(self, space, dim, pq_sub_vectors=8, num_bits_sub_vector=8, dtype='float32', enable_pq=False, is_sparse_graph=False, save_id_vec_map=False):
        self.lock = threading.Lock()
        self.dict_labels = {}
        self.cur_ind = 0
        self.dim = dim  # Original Vector Dimensionality
        self.pq_sub_vectors = pq_sub_vectors
        self.num_bits_sub_vector = num_bits_sub_vector
        self.pq = faiss.ProductQuantizer(dim, pq_sub_vectors, num_bits_sub_vector)
        self.set_dtype(dtype)
        self.enable_pq = enable_pq
        self.is_sparse_graph = is_sparse_graph
        if self.enable_pq:
            self.index = hnswlib.Index(space, pq_sub_vectors)
        else:
            self.index = hnswlib.Index(space, dim)
        self.save_id_vec_map = save_id_vec_map
        self.id_vec_map = {}

    def init_index(self, max_elements, ef_construction=200, M=16):
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)

    def get_test_str(self):
        return "Index loaded"

    def printDebugLog(self, return_payload, isDebug=False, debug_str=''):
        if isDebug:
            return_payload['debug_stats'] = debug_str
            print(debug_str)

    def add_items(self, data_in, ids=None):
        data = self.get_quantized_batch(data_in)
        if ids is not None:
            assert len(data) == len(ids)
        num_added = len(data)
        with self.lock:
            start = self.cur_ind
            self.cur_ind += num_added
        int_labels = []

        if ids is not None:
            for dl in ids:
                int_labels.append(start)
                self.dict_labels[str(start)] = dl
                start += 1
        else:
            for _ in range(len(data)):
                int_labels.append(start)
                self.dict_labels[str(start)] = start
                start += 1
        print(np.asarray(int_labels))
        self.index.add_items(data=data, ids=np.asarray(int_labels))
        if self.save_id_vec_map:
            self.id_vec_map.update({obj["pk"]: embedding for obj, embedding in zip(ids, data_in)})

    def set_ef(self, ef):
        self.index.set_ef(ef)

    def set_dtype(self, dtype_str):
        try:
            # Convert dtype from string to actual NumPy dtype
            self.dtype = np.dtype(dtype_str)
        except TypeError:
            raise ValueError(f"Invalid dtype: {dtype_str}")

    def load_index(self, path, file_name):
        try:
            hnsw_path = f'{path}/{file_name}'
            pq_path = f'{path}/product_quantizer.pq'
            print(f'Loading Index from {hnsw_path} and loading PQ from {pq_path}')
            self.index.load_index(hnsw_path)
            self.pq = faiss.read_ProductQuantizer(f'{path}/product_quantizer.pq')
            with open(hnsw_path + ".pkl", "rb") as f:
                self.cur_ind, self.dict_labels = pickle.load(f)
        except Exception as e:
            print(f"An error occurred loading index: {e}")

    def save_index(self, path, file_name):
        hnsw_path = f'{path}/{file_name}'
        pq_path = f'{path}/product_quantizer.pq'
        print(f'Saving Index to {hnsw_path} and saving PQ to {pq_path}')
        self.index.save_index(hnsw_path)
        faiss.write_ProductQuantizer(self.pq, pq_path)
        with open(hnsw_path + ".pkl", "wb") as f:
            pickle.dump((self.cur_ind, self.dict_labels), f)

    def set_num_threads(self, num_threads):
        self.index.set_num_threads(num_threads)

    def get_current_count(self):
        return self.index.get_current_count()

    def get_quantized_batch(self, input_vecs):
        if self.enable_pq:
            self.pq.train(input_vecs)
            quantized_data = self.pq.compute_codes(input_vecs)  # Step 3: Encode (Quantize) the dataset
            return quantized_data
        else:
            return input_vecs

    def get_quantized_query(self, input_vec, explain_res):
        if self.enable_pq:
            query_vector = np.array([input_vec],
                                    dtype=self.dtype)  # np.random.random((1, d)).astype(np.float32)  # Replace with your actual query vector
            quantized_query = self.pq.compute_codes(query_vector)
            if explain_res:
                print(f'Quantized Vec: {quantized_query[0]}')
            return quantized_query[0]
        else:
            if explain_res:
                print(f'Non Quantized Vec: {input_vec}')
            return input_vec

    def knn_query(self, data_vec, filter_func_name='', filter_params={}, k=1, timeout_ms=100, num_threads=1,
                  explain_res=False):
        try:
            data = self.get_quantized_query(data_vec, explain_res)
            internal_ids_final: list(int) = []
            labels_final_data_obj = []
            distances_final = []
            seen_ids = {}
            passes = 0
            start_time = time.time() * 1000  # Record the start time
            debug_str = ''
            while len(internal_ids_final) < k:
                # Check if the timeout has been reached
                if (time.time() * 1000 - start_time) > timeout_ms:
                    debug_str += f"Timeout ({timeout_ms} ms) reached. Exiting loop. \n"
                    break
                passes = passes + 1
                curr_k = k - len(internal_ids_final)
                start_time1 = time.time()
                if passes == 1 and not self.is_sparse_graph:
                    ## Only on the first pass we fetch a lot more elements than we need, since we expect
                    curr_k = curr_k * 5
                try:
                    labels_ints, distances_int = self.index.knn_query(data=data, k=curr_k, num_threads=num_threads,
                                                                      filter=lambda idx: idx not in seen_ids)
                except Exception as e:
                    ## Break if the K is too high and we are not able to fetch enough elements and get the error
                    ## Cannot return the results in a contiguous 2D array. Probably ef or M is too small
                    debug_str += f"{e} \n"
                    break
                end_time1 = (time.time())
                timedelta = (end_time1 - start_time1) * 1000
                if passes == 1:
                    debug_str += f"TimKNN: {timedelta} elemsFetched:{len(distances_final)} currK:{k} \n"
                else:
                    debug_str += f"TimKNN: {timedelta} elemsFetched:{len(distances_final)} currK:{curr_k} \n"

                labels_int = labels_ints[0]
                distances = distances_int[0]
                # print(len(labels_final_id))
                start_time1 = time.time()
                # Apply filter function
                filter_fun = filter_map.get(filter_func_name, None)
                for i in range(0, len(labels_int)):
                    li = labels_int[i]
                    dis = distances[i]
                    # print(li)
                    ob = self.dict_labels[str(li)]
                    if (filter_fun is None) or (filter_fun(ob, filter_params)):
                        # print(ob)
                        labels_final_data_obj.append(ob)
                        internal_ids_final.append(li)
                        distances_final.append(dis)
                    seen_ids[li] = 1
                end_time1 = (time.time())
                timedelta = (end_time1 - start_time1) * 1000
                debug_str += f"FilterFuncTime: {timedelta} elemsFetched:{len(distances_final)} \n"
                debug_str += f'loops: {passes} \n'
            return_payload = {'result_objs': labels_final_data_obj, 'scores': distances_final}
            self.printDebugLog(return_payload, explain_res, debug_str)

            # Format Results for return
            knn_w_score = [{'id': item['pk'], 'score': return_payload['scores'][i]} for i, item in
                           enumerate(return_payload['result_objs'])]

            # Clip Results to K
            if len(knn_w_score) > k:
                knn_w_score = knn_w_score[:k]
            result = {'knn_w_score': knn_w_score}
            if explain_res:
                result['result_objs'] = return_payload['result_objs']
                result['debug_stats'] = return_payload['debug_stats']
            return result
        except Exception as e:
            print(f"An error occurred: {e}")
            return {'error': f"An error occurred: {e}"}

    def knn_query_by_key(self, key, filter_func_name='', filter_params={}, k=1, timeout_ms=100, num_threads=1,
                  explain_res=False):
        if key not in self.id_vec_map:
            print("Key {} does not exist in current index.".format(key))
            return {}
        return self.knn_query(self.id_vec_map[key], filter_func_name, filter_params, k, timeout_ms, num_threads, explain_res)

    def _filter(self, idx, filter_params, filter_func_name):
        obj = self.dict_labels[str(idx)]
        filter_fun = filter_map.get(filter_func_name)
        return filter_fun(obj, filter_params)

    def knn_query_filter_on_traversal(self, data_vec, filter_func_name='', filter_params={}, k=1, timeout_ms=100, num_threads=1,
                  explain_res=False):
        try:
            filter_fun = filter_map.get(filter_func_name, None)

            if filter_fun is None:
                return {'error': f"filter func name {filter_func_name} does not exist'"}

            data = self.get_quantized_query(data_vec, explain_res)

            while k > 0:
                try:
                    labels_ints, distances_int = self.index.knn_query(
                        data=data,
                        k=k,
                        num_threads=1,
                        filter=lambda idx: self._filter(idx, filter_params, filter_func_name)
                    )

                    knn_w_score = []
                    for ii, score in zip(labels_ints[0], distances_int[0]):
                        append_obj = {'id': self.dict_labels[str(ii)]['pk'], 'score': score}
                        if explain_res:
                            append_obj['result_objs'] = self.dict_labels[str(ii)]
                        knn_w_score.append(append_obj)

                    return {'knn_w_score': knn_w_score}

                except Exception as e:
                    k = int(k / 4) if k > 0 else 0

            return {'knn_w_score': []}

        except Exception as e:
            print(f"An error occurred: {e}")
            return {'error': f"An error occurred: {e}"}