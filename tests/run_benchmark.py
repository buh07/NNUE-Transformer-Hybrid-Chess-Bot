"""
Automated Benchmark Suite
Run standardized tests against Stockfish at multiple strength levels
"""

import sys
import time
from pathlib import Path
from HybridChessBot import HybridChessBot
from play_vs_stockfish import StockfishBot, play_match


def run_benchmark_suite():
    """Run complete benchmark suite"""
    
    print("=" * 70)
    print("Hybrid Bot Benchmark Suite")
    print("=" * 70)
    print()
    
    # Find Stockfish
    stockfish_path = "./Stockfish/src/stockfish"
    if not Path(stockfish_path).exists():
        print(f"ERROR: Stockfish not found at {stockfish_path}")
        sys.exit(1)
    
    # Benchmark configurations
    benchmarks = [
        {
            'name': 'Beginner Test',
            'sf_elo': 1350,
            'hybrid_depth': 4,
            'hybrid_time': 3.0,
            'games': 4,
            'description': 'Should win most games'
        },
        {
            'name': 'Intermediate Test',
            'sf_elo': 1650,
            'hybrid_depth': 5,
            'hybrid_time': 5.0,
            'games': 4,
            'description': 'Should be competitive'
        },
        {
            'name': 'Advanced Test',
            'sf_elo': 1850,
            'hybrid_depth': 6,
            'hybrid_time': 8.0,
            'games': 4,
            'description': 'Challenging match'
        }
    ]
    
    # Run each benchmark
    all_results = []
    
    for i, config in enumerate(benchmarks, 1):
        print(f"\n{'#'*70}")
        print(f"Benchmark {i}/{len(benchmarks)}: {config['name']}")
        print(f"{'#'*70}")
        print(f"Description: {config['description']}")
        print(f"Stockfish ELO: {config['sf_elo']}")
        print(f"Hybrid: Depth {config['hybrid_depth']}, Time {config['hybrid_time']}s")
        print(f"Games: {config['games']}")
        print()
        
        input(f"Press Enter to start {config['name']}...")
        
        # Create bots
        print("\nInitializing bots...")
        hybrid = HybridChessBot(
            checkpoint='checkpoints/best_phase2.pt',
            depth=config['hybrid_depth'],
            time_limit=config['hybrid_time'],
            verbose=False
        )
        
        stockfish = StockfishBot(
            stockfish_path=stockfish_path,
            elo=config['sf_elo'],
            time_limit=1.0
        )
        
        print("Bots ready!\n")
        
        # Play match
        try:
            start_time = time.time()
            results = play_match(
                hybrid_bot=hybrid,
                stockfish_bot=stockfish,
                num_games=config['games'],
                hybrid_name="HybridBot",
                stockfish_name=f"SF-{config['sf_elo']}"
            )
            elapsed = time.time() - start_time
            
            # Add metadata
            results['benchmark_name'] = config['name']
            results['sf_elo'] = config['sf_elo']
            results['hybrid_depth'] = config['hybrid_depth']
            results['hybrid_time'] = config['hybrid_time']
            results['total_time'] = elapsed
            
            all_results.append(results)
            
            print(f"\nBenchmark completed in {elapsed/60:.1f} minutes")
            
        finally:
            stockfish.quit()
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUITE SUMMARY")
    print("=" * 70)
    print()
    
    total_games = 0
    total_score = 0
    total_possible = 0
    
    for result in all_results:
        name = result['benchmark_name']
        score = result['hybrid_score']
        total = result['wins'] + result['losses'] + result['draws']
        win_rate = score / total * 100
        
        print(f"{name}:")
        print(f"  Opponent: Stockfish ELO {result['sf_elo']}")
        print(f"  Result: {result['wins']}W - {result['losses']}L - {result['draws']}D")
        print(f"  Score: {score}/{total} ({win_rate:.1f}%)")
        print(f"  Time: {result['total_time']/60:.1f} minutes")
        print()
        
        total_games += total
        total_score += score
        total_possible += total
    
    overall_rate = total_score / total_possible * 100
    print(f"Overall Performance:")
    print(f"  Total games: {total_games}")
    print(f"  Total score: {total_score}/{total_possible} ({overall_rate:.1f}%)")
    
    # Estimate strength
    print(f"\nStrength Estimation:")
    if overall_rate >= 70:
        print(f"  Estimated ELO: 1700-1900 (Strong intermediate)")
    elif overall_rate >= 55:
        print(f"  Estimated ELO: 1550-1700 (Intermediate)")
    elif overall_rate >= 40:
        print(f"  Estimated ELO: 1400-1550 (Improving)")
    else:
        print(f"  Estimated ELO: <1400 (Beginner)")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("Hybrid Bot Benchmark Results\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        for result in all_results:
            f.write(f"{result['benchmark_name']}\n")
            f.write(f"  Stockfish ELO: {result['sf_elo']}\n")
            f.write(f"  Hybrid Config: Depth {result['hybrid_depth']}, Time {result['hybrid_time']}s\n")
            f.write(f"  Games: {result['wins']}W - {result['losses']}L - {result['draws']}D\n")
            f.write(f"  Score: {result['hybrid_score']}/{result['wins'] + result['losses'] + result['draws']}\n")
            f.write(f"  Win Rate: {result['hybrid_score']/(result['wins'] + result['losses'] + result['draws'])*100:.1f}%\n")
            f.write("\n")
        
        f.write(f"\nOverall: {total_score}/{total_possible} ({overall_rate:.1f}%)\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print("\nBenchmark suite complete!")


if __name__ == '__main__':
    run_benchmark_suite()
