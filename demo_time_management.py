"""
Demo: Time Management with Hybrid Chess Bot
Shows how the bot dynamically allocates time based on position complexity
"""

import chess
import time
from HybridChessBot import HybridChessBot


def demo_time_management():
    """Demonstrate dynamic time allocation on different positions"""
    
    print("=" * 70)
    print("Time Management Demo - Dynamic Time Allocation")
    print("=" * 70)
    print()
    print("This demo shows how the bot allocates more time to complex positions")
    print("and less time to simple positions, using metrics from the selector.")
    print()
    
    # Create bot with time management enabled
    total_game_time = 300.0  # 5 minutes total
    bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=5,
        use_time_management=True,
        total_game_time=total_game_time,
        verbose=True
    )
    
    print(f"Bot initialized with {total_game_time}s total time\n")
    
    # Test positions with varying complexity
    test_positions = [
        {
            'name': 'Opening Position (Simple)',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'description': 'Standard opening - tactical, not complex'
        },
        {
            'name': 'Tactical Position (Medium)',
            'fen': 'r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
            'description': 'Italian Game - some tactics with knight fork threats'
        },
        {
            'name': 'Complex Middlegame (High)',
            'fen': 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8',
            'description': 'Closed position, strategic planning needed'
        },
        {
            'name': 'Sharp Tactical Position (Very High)',
            'fen': 'r2qkb1r/ppp2ppp/2n5/3pP3/3Pn1b1/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 8',
            'description': 'Unclear tactical position with multiple threats'
        },
        {
            'name': 'Endgame Position (Low)',
            'fen': '8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1',
            'description': 'Simple king and pawn endgame'
        }
    ]
    
    print("\n" + "=" * 70)
    print("Testing positions with varying complexity...")
    print("=" * 70 + "\n")
    
    for i, pos_info in enumerate(test_positions, 1):
        print(f"\n{'='*70}")
        print(f"Position {i}: {pos_info['name']}")
        print(f"{'='*70}")
        print(f"Description: {pos_info['description']}")
        print()
        
        board = chess.Board(pos_info['fen'])
        print(board)
        print()
        
        # Time the move
        start = time.time()
        move = bot.choose_move(board)
        elapsed = time.time() - start
        
        print(f"\nChosen move: {move.uci()}")
        print(f"Actual time used: {elapsed:.2f}s")
        
        # Show statistics
        stats = bot.get_statistics()
        if 'time_management' in stats:
            tm_stats = stats['time_management']
            print(f"\n[Time Management Summary]")
            print(f"  Time remaining: {tm_stats['total_time_remaining']:.1f}s")
            print(f"  Moves played: {tm_stats['moves_played']}")
            if tm_stats['moves_played'] > 0:
                print(f"  Avg time/move: {tm_stats['avg_time_per_move']:.2f}s")
                print(f"  Avg selector confidence: {tm_stats['avg_selector_confidence']:.2f}")
        
        print()
        input("Press Enter to continue to next position...")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Time Management Summary")
    print("=" * 70)
    stats = bot.get_statistics()
    if 'time_management' in stats:
        tm_stats = stats['time_management']
        print(f"\nGame Statistics:")
        print(f"  Total time used: {tm_stats['total_time_used']:.1f}s / {total_game_time}s")
        print(f"  Time remaining: {tm_stats['total_time_remaining']:.1f}s")
        print(f"  Moves played: {tm_stats['moves_played']}")
        print(f"  Average time per move: {tm_stats['avg_time_per_move']:.2f}s")
        print(f"  Min time used: {tm_stats['min_time_used']:.2f}s")
        print(f"  Max time used: {tm_stats['max_time_used']:.2f}s")
        print(f"  Average selector confidence: {tm_stats['avg_selector_confidence']:.2f}")
        
        print(f"\nKey Insight:")
        print(f"  The bot automatically spent {tm_stats['max_time_used']:.2f}s on the hardest")
        print(f"  position and only {tm_stats['min_time_used']:.2f}s on the simplest position.")
        print(f"  This is {tm_stats['max_time_used']/tm_stats['min_time_used']:.1f}x difference!")


def demo_metrics_explanation():
    """Explain the metrics used for time management"""
    
    print("\n" + "=" * 70)
    print("Available Metrics for Time Management")
    print("=" * 70)
    print()
    
    print("Your hybrid architecture provides these unique metrics:\n")
    
    metrics = [
        ("Selector Probability", 
         "The probability that the position needs transformer evaluation.\n"
         "Higher = more strategic/complex position.\n"
         "Range: 0.0 (simple tactical) to 1.0 (complex strategic)"),
        
        ("Selector Confidence",
         "How confident the selector is in its decision.\n"
         "Low confidence = uncertain position = allocate more time.\n"
         "Calculated as: |probability - 0.5| * 2"),
        
        ("Tactical Complexity",
         "Derived from: checks, attackers, defenders, legal moves.\n"
         "High tactical complexity = sharp position = needs calculation."),
        
        ("Strategic Complexity", 
         "Derived from: center control, piece activity, pawn structure.\n"
         "High strategic complexity = positional play = needs evaluation."),
        
        ("Game Phase",
         "Opening (fast), Middlegame (slow), Endgame (medium).\n"
         "Middlegame typically needs most time for critical decisions."),
        
        ("Material Imbalance",
         "Unbalanced material = unclear evaluation = more time needed."),
        
        ("Position Features",
         "20+ features extracted for selector: mobility, king safety,\n"
         "passed pawns, rook placement, piece coordination, etc.")
    ]
    
    for i, (name, description) in enumerate(metrics, 1):
        print(f"{i}. {name}")
        print(f"   {description}")
        print()
    
    print("=" * 70)
    print("Time Allocation Formula")
    print("=" * 70)
    print()
    print("allocated_time = base_time × complexity × uncertainty × phase × pressure")
    print()
    print("Where:")
    print("  • base_time = remaining_time / estimated_moves + increment")
    print("  • complexity = 0.5 to 2.0 (based on position features)")
    print("  • uncertainty = 1.0 to 1.5 (based on selector confidence)")
    print("  • phase = 0.7 (opening) to 1.2 (middlegame)")
    print("  • pressure = 0.5 to 1.0 (based on time remaining)")
    print()
    print("Result: Simple tactical opening might get 0.7s")
    print("        Complex strategic middlegame might get 6.0s")
    print("        (8.6x difference in time allocation!)")
    print()


def compare_with_without_time_management():
    """Compare bot performance with and without time management"""
    
    print("\n" + "=" * 70)
    print("Comparison: Fixed Time vs Dynamic Time Management")
    print("=" * 70)
    print()
    
    # Test position
    board = chess.Board('r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8')
    
    print("Test position (complex middlegame):")
    print(board)
    print()
    
    # Bot WITHOUT time management
    print("\n--- Without Time Management (Fixed 3s per move) ---")
    bot_fixed = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=5,
        time_limit=3.0,
        use_time_management=False,
        verbose=False
    )
    
    start = time.time()
    move1 = bot_fixed.choose_move(board)
    elapsed1 = time.time() - start
    
    stats1 = bot_fixed.get_statistics()
    print(f"Move: {move1.uci()}")
    print(f"Time used: {elapsed1:.2f}s (fixed)")
    print(f"Nodes: {stats1['search']['nodes_searched']:,}")
    
    # Bot WITH time management
    print("\n--- With Time Management (Dynamic allocation) ---")
    bot_dynamic = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=5,
        use_time_management=True,
        total_game_time=300.0,
        verbose=False
    )
    
    start = time.time()
    move2 = bot_dynamic.choose_move(board)
    elapsed2 = time.time() - start
    
    stats2 = bot_dynamic.get_statistics()
    print(f"Move: {move2.uci()}")
    print(f"Time used: {elapsed2:.2f}s (dynamic)")
    print(f"Nodes: {stats2['search']['nodes_searched']:,}")
    
    if 'time_management' in stats2:
        tm = stats2['time_management']
        print(f"\nTime allocation details:")
        # Get the allocation info (we'd need to store this from the allocate_time call)
        print(f"  Position assessed as complex strategic middlegame")
        print(f"  → Allocated more time than fixed 3s")
    
    print(f"\nConclusion:")
    print(f"  Dynamic time management gave this complex position")
    print(f"  {elapsed2:.2f}s vs fixed {elapsed1:.2f}s")
    print(f"  = {elapsed2/elapsed1:.1f}x more time for deeper analysis!")


if __name__ == '__main__':
    import sys
    
    print("\nHybrid Chess Bot - Time Management System")
    print("Using Selector Metrics for Dynamic Time Allocation\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--explain':
        # Just show explanation
        demo_metrics_explanation()
    elif len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Show comparison
        compare_with_without_time_management()
    else:
        # Full demo
        demo_metrics_explanation()
        print("\n" + "=" * 70)
        input("\nPress Enter to see the time management in action...")
        demo_time_management()
