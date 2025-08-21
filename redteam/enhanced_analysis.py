"""Enhanced analysis functions for conversation and Harmony format support."""

import numpy as np
from typing import List, Dict, Any

def analyze_conversation_candidates(conversation_candidates: List, top_k: int = 5):
    """Deep analysis of conversation candidates"""
    if not conversation_candidates:
        print("No conversation candidates to analyze")
        return
    
    print(f"\n🗣️  Detailed Conversation Analysis ({len(conversation_candidates)} candidates)")
    print("=" * 80)
    
    # Sort by cumulative score
    def get_conv_score(conv):
        if hasattr(conv, 'cumulative_score'):
            return conv.cumulative_score
        elif isinstance(conv, dict) and 'cumulative_score' in conv:
            return conv['cumulative_score']
        return 0.0
    
    sorted_conversations = sorted(conversation_candidates, key=get_conv_score, reverse=True)
    
    for i, conv in enumerate(sorted_conversations[:top_k], 1):
        score = get_conv_score(conv)
        turns = getattr(conv, 'turns', [])
        contradictions = getattr(conv, 'contradictions', [])
        consistency = getattr(conv, 'consistency_score', 1.0)
        
        print(f"\n#{i} Conversation | Score: {score:.3f} | Turns: {len(turns)} | Consistency: {consistency:.3f}")
        print(f"Contradictions Found: {len(contradictions)}")
        
        # Show contradictions
        for j, contradiction in enumerate(contradictions[:3], 1):  # Show first 3
            if isinstance(contradiction, dict):
                turn1 = contradiction.get('turn1', 'N/A')
                turn2 = contradiction.get('turn2', 'N/A')
                contra_type = contradiction.get('type', 'unknown')
                print(f"   {j}. {contra_type}: Turn {turn1} vs Turn {turn2}")
        
        # Show conversation flow
        print("   Conversation Flow:")
        for turn_num, turn in enumerate(turns[:3], 1):  # Show first 3 turns
            prompt = turn.get('prompt', '')[:50]
            response = turn.get('response', '')[:100]
            print(f"   Turn {turn_num}: {prompt}... → {response}...")
        
        if len(turns) > 3:
            print(f"   ... and {len(turns) - 3} more turns")
        
        print("-" * 80)

def harmony_format_analysis(candidates: List[Dict[str, Any]]):
    """Detailed analysis of Harmony format usage"""
    print("\n🧠 Harmony Format Analysis")
    print("=" * 60)
    
    thinking_responses = []
    
    for candidate in candidates:
        judge_result = candidate.get('judge_result', {})
        harmony_sections = judge_result.get('harmony_sections', {})
        
        if harmony_sections:
            thinking = harmony_sections.get('thinking', '').strip()
            output = harmony_sections.get('output', '').strip()
            
            if thinking:
                thinking_responses.append({
                    'candidate': candidate,
                    'thinking': thinking,
                    'output': output,
                    'score': candidate.get('final_score', 0.0)
                })
    
    if not thinking_responses:
        print("No Harmony thinking content detected")
        return
    
    print(f"Found {len(thinking_responses)} responses with thinking content:")
    
    # Sort by score
    thinking_responses.sort(key=lambda x: x['score'], reverse=True)
    
    for i, item in enumerate(thinking_responses[:5], 1):  # Show top 5
        print(f"\n#{i} | Score: {item['score']:.3f}")
        print(f"Thinking: \"{item['thinking'][:200]}...\"")
        print(f"Output: \"{item['output'][:200]}...\"")
        print("-" * 60)

def conversation_contradiction_report(conversation_candidates: List):
    """Generate detailed contradiction analysis report"""
    print("\n🔍 Conversation Contradiction Analysis")
    print("=" * 70)
    
    all_contradictions = []
    
    for conv in conversation_candidates:
        contradictions = getattr(conv, 'contradictions', [])
        conv_id = getattr(conv, 'conversation_id', 'unknown')
        
        for contradiction in contradictions:
            if isinstance(contradiction, dict):
                contradiction['conversation_id'] = conv_id
                all_contradictions.append(contradiction)
    
    if not all_contradictions:
        print("No contradictions found in conversation data")
        return
    
    print(f"Total contradictions found: {len(all_contradictions)}")
    
    # Group by type
    by_type = {}
    for contradiction in all_contradictions:
        contra_type = contradiction.get('type', 'unknown')
        if contra_type not in by_type:
            by_type[contra_type] = []
        by_type[contra_type].append(contradiction)
    
    print("\nContradiction Types:")
    for contra_type, contras in by_type.items():
        print(f"   {contra_type}: {len(contras)} instances")
        
        # Show example
        if contras:
            example = contras[0]
            turn1 = example.get('turn1', 'N/A')
            turn2 = example.get('turn2', 'N/A')
            print(f"      Example: Turn {turn1} vs Turn {turn2}")
    
    print("-" * 70)

def enhanced_export_analysis_report(candidates: List[Dict[str, Any]], debug_info: Dict[str, Any], 
                                   filename: str = "enhanced_red_team_analysis.txt"):
    """Export comprehensive analysis including conversation and Harmony data"""
    conversation_candidates = debug_info.get('conversation_candidates', [])
    harmony_analysis = debug_info.get('harmony_analysis', [])
    
    with open(filename, 'w') as f:
        f.write("ENHANCED RED-TEAMING ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary statistics
        scores = debug_info.get('scores_history', [])
        if scores:
            f.write(f"Total attempts: {len(scores)}\n")
            f.write(f"Max score: {max(scores):.3f}\n")
            f.write(f"Mean score: {np.mean(scores):.3f}\n")
            f.write(f"Score std dev: {np.std(scores):.3f}\n\n")
        
        # Conversation analysis
        if conversation_candidates:
            f.write("CONVERSATION ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total conversations: {len(conversation_candidates)}\n")
            
            total_contradictions = sum(len(getattr(c, 'contradictions', [])) for c in conversation_candidates)
            f.write(f"Total contradictions: {total_contradictions}\n")
            
            avg_consistency = np.mean([getattr(c, 'consistency_score', 1.0) for c in conversation_candidates])
            f.write(f"Average consistency: {avg_consistency:.3f}\n\n")
        
        # Harmony format analysis
        if harmony_analysis:
            f.write("HARMONY FORMAT ANALYSIS\n")
            f.write("-" * 35 + "\n")
            thinking_count = sum(1 for h in harmony_analysis if h.get('thinking_detected', False))
            f.write(f"Responses with thinking: {thinking_count}/{len(harmony_analysis)}\n")
            f.write(f"Thinking detection rate: {thinking_count/len(harmony_analysis)*100:.1f}%\n\n")
        
        # Top candidates
        f.write("TOP 10 CANDIDATES\n")
        f.write("-" * 20 + "\n")
        
        def get_score(candidate):
            if 'final_score' in candidate:
                return candidate['final_score']
            return candidate.get('judge_result', {}).get('score', 0.0)
        
        sorted_candidates = sorted(candidates, key=get_score, reverse=True)
        for i, candidate in enumerate(sorted_candidates[:10], 1):
            score = get_score(candidate)
            candidate_type = candidate.get('type', 'single_turn')
            f.write(f"{i}. Score: {score:.3f} | Type: {candidate_type} | Family: {candidate.get('family', 'Unknown')}\n")
            
            if candidate_type == 'conversation':
                conv_candidate = candidate.get('conversation_candidate')
                if conv_candidate and hasattr(conv_candidate, 'turns'):
                    f.write(f"   Turns: {len(conv_candidate.turns)} | Contradictions: {len(getattr(conv_candidate, 'contradictions', []))}\n")
            else:
                response = candidate.get('response', 'No response')[:100]
                f.write(f"   Response: {response}...\n")
            f.write("\n")
    
    print(f"Enhanced analysis report exported to {filename}")