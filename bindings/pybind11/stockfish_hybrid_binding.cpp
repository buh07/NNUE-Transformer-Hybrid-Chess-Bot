#include <algorithm>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bitboard.h"
#include "evaluate.h"
#include "movegen.h"
#include "nnue/features/full_threats.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/nnue_common.h"
#include "position.h"
#include "uci.h"

namespace py = pybind11;
namespace sf = Stockfish;
namespace nn = Stockfish::Eval::NNUE;

namespace {

constexpr const char* kStartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

void ensure_stockfish_initialized() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        sf::Bitboards::init();
        sf::Position::init();
        sf::Eval::NNUE::Features::init_threat_offsets();
    });
}

nn::EvalFile make_eval_file(const std::string& default_name, const std::string& current) {
    nn::EvalFile eval_file{};
    eval_file.defaultName = default_name.c_str();
    eval_file.current     = current.empty() ? default_name.c_str() : current.c_str();
    eval_file.netDescription = "pybind11 binding";
    return eval_file;
}

int piece_value(sf::PieceType pt) {
    switch (pt) {
    case sf::PAWN:
        return 100;
    case sf::KNIGHT:
        return 320;
    case sf::BISHOP:
        return 330;
    case sf::ROOK:
        return 500;
    case sf::QUEEN:
        return 900;
    default:
        return 0;
    }
}

}  // namespace

struct BindingTTEntry {
    double score;
    int    depth;
    int    flag;
    sf::Move move;
};

class StockfishHybridEngine {
   public:
    StockfishHybridEngine(const std::string& binary_dir,
                          const std::string& big_net,
                          const std::string& small_net,
                          bool              chess960)
        : binary_dir_(binary_dir), chess960_(chess960), states_(1) {
        ensure_stockfish_initialized();
        load_networks_internal(big_net, small_net);
        pos_.set(kStartFEN, chess960_, &states_.back());
    }

    void load_networks(const std::string& big_net, const std::string& small_net) {
        load_networks_internal(big_net, small_net);
    }

    void set_fen(const std::string& fen,
                 bool               chess960,
                 const std::vector<std::string>& moves) {
        const std::string fen_to_set =
          fen.empty() || fen == "startpos" ? std::string(kStartFEN) : fen;

        chess960_ = chess960;
        states_.clear();
        states_.emplace_back();
        move_history_.clear();
        pos_.set(fen_to_set, chess960_, &states_.back());

        for (const auto& mv : moves)
            push_move(mv);
    }

    void push_move(const std::string& move_uci) {
        auto move = sf::UCIEngine::to_move(pos_, move_uci);
        if (move == sf::Move::none())
            throw std::invalid_argument("Illegal or unknown move: " + move_uci);

        states_.emplace_back();
        pos_.do_move(move, states_.back());
        move_history_.push_back(move);
    }

    void pop_move() {
        if (move_history_.empty())
            throw std::runtime_error("No moves to undo");

        auto move = move_history_.back();
        move_history_.pop_back();
        pos_.undo_move(move);
        states_.pop_back();
    }

    double evaluate(bool white_pov) {
        if (!networks_)
            throw std::runtime_error("Networks not loaded");

        nn::AccumulatorStack accumulators;
        accumulators.reset();

        if (!caches_)
            caches_ = std::make_unique<nn::AccumulatorCaches>(*networks_);

        sf::Value value =
          sf::Eval::evaluate(*networks_, pos_, accumulators, *caches_, sf::VALUE_ZERO);

        if (white_pov && pos_.side_to_move() == sf::BLACK)
            value = -value;

        return static_cast<double>(sf::UCIEngine::to_cp(value, pos_));
    }

    std::string fen() const { return pos_.fen(); }

    std::vector<std::string> ordered_moves() {
        ensure_stockfish_initialized();
        sf::MoveList<sf::LEGAL> moves(pos_);
        struct ScoredMove {
            sf::Move move;
            int      score;
        };
        std::vector<ScoredMove> scored;
        scored.reserve(moves.size());

        for (const auto& m : moves) {
            int score = 0;
            const auto to   = sf::to_sq(m);
            const auto from = sf::from_sq(m);

            const bool is_capture = pos_.piece_on(to) != sf::NO_PIECE;
            if (is_capture) {
                const auto victim   = sf::type_of(pos_.piece_on(to));
                const auto attacker = pos_.piece_on(from) == sf::NO_PIECE
                                        ? sf::NO_PIECE_TYPE
                                        : sf::type_of(pos_.piece_on(from));
                score += 1000 + 10 * piece_value(victim) - piece_value(attacker);
            }

            if (sf::type_of(m) == sf::PROMOTION)
                score += 800 + piece_value(sf::promotion_type(m));

            if (pos_.gives_check(m))
                score += 200;

            scored.push_back({m, score});
        }

        std::sort(scored.begin(),
                  scored.end(),
                  [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });

        std::vector<std::string> out;
        out.reserve(scored.size());
        for (const auto& s : scored)
            out.emplace_back(sf::UCIEngine::move_to_uci(pos_, s.move));
        return out;
    }

    py::object probe_tt() {
        const auto key = pos_.key();
        const auto it  = tt_store_.find(key);
        if (it == tt_store_.end())
            return py::none();

        py::dict entry;
        entry["score"] = it->second.score;
        entry["depth"] = it->second.depth;
        entry["flag"]  = it->second.flag;
        entry["move"]  = it->second.move == sf::Move::none()
                           ? ""
                           : sf::UCIEngine::move_to_uci(pos_, it->second.move);
        return entry;
    }

    void store_tt(double score, int depth, int flag, const std::string& move_uci) {
        BindingTTEntry entry;
        entry.score = score;
        entry.depth = depth;
        entry.flag  = flag;
        entry.move  = move_uci.empty() ? sf::Move::none() : sf::UCIEngine::to_move(pos_, move_uci);
        tt_store_[pos_.key()] = entry;
    }

   private:
    void load_networks_internal(const std::string& big_net, const std::string& small_net) {
        auto big_file   = make_eval_file(EvalFileDefaultNameBig, big_net);
        auto small_file = make_eval_file(EvalFileDefaultNameSmall, small_net);

        auto big_network =
          std::make_unique<nn::NetworkBig>(big_file, nn::EmbeddedNNUEType::BIG);
        auto small_network =
          std::make_unique<nn::NetworkSmall>(small_file, nn::EmbeddedNNUEType::SMALL);

        big_network->load(binary_dir_, big_net);
        small_network->load(binary_dir_, small_net);

        networks_ =
          std::make_unique<nn::Networks>(std::move(big_network), std::move(small_network));
        caches_ = std::make_unique<nn::AccumulatorCaches>(*networks_);
    }

    std::string                           binary_dir_;
    bool                                  chess960_;
    sf::Position                          pos_;
    std::deque<sf::StateInfo>             states_;
    std::vector<sf::Move>                 move_history_;
    std::unique_ptr<nn::Networks>         networks_;
    std::unique_ptr<nn::AccumulatorCaches> caches_;
    std::unordered_map<sf::Key, BindingTTEntry> tt_store_;
};

PYBIND11_MODULE(stockfish_hybrid_binding, m) {
    m.doc() = "Stockfish hybrid pybind11 bindings (evaluation API)";

    py::class_<StockfishHybridEngine>(m, "StockfishHybridEngine")
      .def(py::init<const std::string&, const std::string&, const std::string&, bool>(),
           py::arg("binary_dir") = "",
           py::arg("big_net") = EvalFileDefaultNameBig,
           py::arg("small_net") = EvalFileDefaultNameSmall,
           py::arg("chess960") = false)
      .def("load_networks", &StockfishHybridEngine::load_networks,
           py::arg("big_net") = EvalFileDefaultNameBig,
           py::arg("small_net") = EvalFileDefaultNameSmall,
           "Reload NNUE networks from disk.")
      .def("set_fen", &StockfishHybridEngine::set_fen,
           py::arg("fen") = kStartFEN,
           py::arg("chess960") = false,
           py::arg("moves") = std::vector<std::string>{},
           "Set the current position (optionally applying a list of moves).")
      .def("push_move", &StockfishHybridEngine::push_move, py::arg("uci_move"))
      .def("pop_move", &StockfishHybridEngine::pop_move)
      .def("evaluate", &StockfishHybridEngine::evaluate,
           py::arg("white_pov") = false,
           "Return evaluation in centipawns (positive = good for the requested perspective).")
      .def("fen", &StockfishHybridEngine::fen)
      .def("ordered_moves", &StockfishHybridEngine::ordered_moves,
           "Return Stockfish-ordered legal moves for the current position.")
      .def("tt_probe", &StockfishHybridEngine::probe_tt,
           "Probe the experimental transposition table for the current position.")
      .def("tt_store",
           &StockfishHybridEngine::store_tt,
           py::arg("score"),
           py::arg("depth"),
           py::arg("flag"),
           py::arg("move"),
           "Store a transposition-table entry for the current position.");
}
