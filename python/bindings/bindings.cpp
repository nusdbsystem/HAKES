#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include <search-worker/workerImpl.h>

struct SearchResult {
  py::array_t<float> scores;
  py::array_t<int64_t> ids;
  std::string msg;
};

// wrapper class around the underlying search worker implementation
class Searcher {
 public:
  Searcher() = default;
  ~Searcher() { w_.Close(); }

  bool Initialize(const std::string& path) { return w_.Initialize(path); }

  bool HasLoadedCollection(const std::string collection_name) {
    return w_.HasLoadedCollection(collection_name);
  }

  std::vector<std::string> ListCollections() {
    return w_.ListCollectionsInternal();
  }

  bool LoadCollection(const std::string& collection_name) {
    return w_.LoadCollectionInternal(collection_name, 0);
  }

  std::string AddwithIds(const std::string& collection_name,
                         py::array_t<float> vecs, py::array_t<int64_t> ids) {
    py::buffer_info ids_buf = ids.request();
    py::buffer_info vecs_buf = vecs.request();
    if (ids_buf.ndim != 1) return "error: ids should have one dimension.";
    int n = ids_buf.shape[0];
    int d = 1;
    if (n == 1) {
      if (vecs_buf.ndim != 1 && vecs_buf.shape[0] != 1) return "error: ids and vecs count inconsistent";
      d = vecs_buf.shape[0];
    } else {
      if (vecs_buf.ndim != 2 || vecs_buf.shape[0] != n) return "error: ids and vecs count inconsistent";
      d = vecs_buf.shape[1];
    }

    float* f_vecs = static_cast<float*>(vecs_buf.ptr);
    int64_t* i_ids = static_cast<int64_t*>(ids_buf.ptr);
    std::string error_msg;
    auto success = w_.AddWithIdsInternal(collection_name, n, d, f_vecs, i_ids, false, &error_msg);
    return (success) ? "success" : "error: " + std::move(error_msg); 
  }

  SearchResult Search(const std::string& collection_name,
                      py::array_t<float> query, int k, const std::string metric,
                      int k_factor, int nprobe) {
    py::buffer_info query_buf = query.request();
    int n = 1;
    int d = 1;
    if (query_buf.ndim == 1) {
      d = query_buf.shape[0];
    } else {
      n = query_buf.shape[0];
      d = query_buf.shape[1];
    }

    float* f_query = static_cast<float*>(query_buf.ptr);
    if (metric != "ip" && metric != "l2") {
      return {};
    }
    faiss::HakesSearchParams params{
        .nprobe = nprobe,
        .k = k,
        .k_factor = k_factor,
        .metric_type = faiss::MetricType((metric == "ip") ? 0 : 1),
    };
    std::unique_ptr<float[]> base_scores;
    std::unique_ptr<faiss::idx_t[]> base_ids;
    std::string error_msg;
    auto success = w_.SearchInternal(collection_name, n, d, f_query, params,
                                     &base_scores, &base_ids, &error_msg);
    if (!success) return {.msg = error_msg};
    std::unique_ptr<float[]> scores;
    std::unique_ptr<faiss::idx_t[]> ids;
    std::vector<faiss::idx_t> k_base_count(n, k);
    success = w_.RerankInternal(collection_name, n, d, f_query, k,
                                k_base_count.data(), base_ids.get(),
                                base_scores.get(), &scores, &ids, &error_msg);
    if (!success) return {.msg = error_msg};

    return {.scores = py::array_t<float>({n, k}, scores.get()),
            .ids = py::array_t<int64_t>({n, k}, ids.get()),
            .msg = error_msg};
  }

  std::string Checkpoint(const std::string& collection_name) {
    std::string error_msg;
    auto success = w_.CheckpointInternal(collection_name, &error_msg);
    return (success) ? "success" : "error: " + std::move(error_msg); 
  }

 private:
  search_worker::WorkerImpl w_;
};

PYBIND11_MODULE(_hakes, m) {
  m.doc() = R"pbdoc(
        HAKES cpp bindings
        -----------------------

        .. currentmodule:: _hakes

        .. autosummary::
           :toctree: _generate

           search_worker::init_params
           search_worker::Worker           
    )pbdoc";

  py::class_<SearchResult>(m, "SearchResult")
      .def(py::init<>())
      .def_readwrite("scores", &SearchResult::scores)
      .def_readwrite("ids", &SearchResult::ids)
      .def_readwrite("msg", &SearchResult::msg);

  py::class_<Searcher>(m, "Searcher")
      .def(py::init<>())
      .def("initialize",
           py::overload_cast<const std::string&>(&Searcher::Initialize))
      .def("list_collections", &Searcher::ListCollections)
      .def("has_loaded_collection", &Searcher::HasLoadedCollection)
      .def("load_collection", &Searcher::LoadCollection)
      .def("add", &Searcher::AddwithIds)
      .def("search", &Searcher::Search)
      .def("checkpoint", &Searcher::Checkpoint);
      

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
