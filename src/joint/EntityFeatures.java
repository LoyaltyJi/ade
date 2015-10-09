package joint;


import java.util.List;






import pipeline.PerceptronEntity;
import cn.fox.utils.StopWord;
import cn.fox.machine_learning.Perceptron;
import cn.fox.machine_learning.PerceptronFeatureFunction;
import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.machine_learning.PerceptronStatus;
import cn.fox.stanford.StanfordTree;
import cn.fox.utils.CharCode;
import cn.fox.utils.WordNetUtil;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.LexicalPattern;
import drug_side_effect_utils.Tool;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.Tree;
import gnu.trove.TObjectDoubleHashMap;
import pipeline.PerceptronEntityOutputData;



public class EntityFeatures extends PerceptronFeatureFunction {
	public EntityFeatures(Perceptron perceptron) {
		super(perceptron);
	}
	
	
	@Override
	public void compute(PerceptronInputData x, PerceptronStatus status, PerceptronOutputData y, Object other,
			TObjectDoubleHashMap<String> map) {
		PerceptronInputData1 input = (PerceptronInputData1)x;
		PerceptronOutputData1 output= (PerceptronOutputData1)y;
		
		Tool tool = (Tool)other;
		
		
		int lastSegmentIndex = output.getLastSegmentIndex(status.tokenIndex);
		Entity lastSegment = output.segments.get(lastSegmentIndex);

		CoreLabel head = AdeJointMain.getHeadWordOfSegment(lastSegment, input);
		String hdLemmaLow = head.lemma().toLowerCase();
		{
			// baseline
			String text = lastSegment.text.toLowerCase();
			addFeature("E#WD_"+lastSegment.type+"_"+text, 1.0, status, y, map);
			
			if(lastSegment.start>=1) {
				addFeature("E#PRETK_"+lastSegment.type+"_"+input.sentInfo.tokens.get(lastSegment.start-1).lemma().toLowerCase(), 1.0, status, y, map);
				addFeature("E#PREPOS_"+lastSegment.type+"_"+input.sentInfo.tokens.get(lastSegment.start-1).tag(), 1.0, status, y, map);
			}
			if(lastSegment.end<=input.sentInfo.tokens.size()-2) {
				addFeature("E#NEXTTK_"+lastSegment.type+"_"+input.sentInfo.tokens.get(lastSegment.end+1).lemma().toLowerCase(), 1.0, status, y, map);
				addFeature("E#NEXTPOS_"+lastSegment.type+"_"+input.sentInfo.tokens.get(lastSegment.end+1).tag(), 1.0, status, y, map);
			}
			
			// head
			int len = hdLemmaLow.length()>4 ? 4:hdLemmaLow.length();
			addFeature("E#HEAD_"+lastSegment.type+"_"+hdLemmaLow, 1.0, status, y, map);
			addFeature("E#PREF_"+lastSegment.type+"_"+hdLemmaLow.substring(0, len), 1.0, status, y, map);
			addFeature("E#SUF_"+lastSegment.type+"_"+hdLemmaLow.substring(hdLemmaLow.length()-len, hdLemmaLow.length()),1.0, status, y, map);
			addFeature("E#HDPOS_"+lastSegment.type+"_"+head.tag(), 1.0, status, y, map);
			
			// pattern
			String posSeq = EntityFeatures.segmentToPosSequence(lastSegment, input);
			addFeature("E#POSSEQ_"+lastSegment.type+"_"+posSeq, 1.0, status, y, map);

			
			
			LexicalPattern lpattern = new LexicalPattern();
			lpattern.getAll(lastSegment.text);
			if(lpattern.ctUpCase == lastSegment.text.length())
				addFeature("E#UCASE_"+lastSegment.type, 1.0, status, y, map);
			else if(lpattern.ctUpCase == 0)
				addFeature("E#LCASE_"+lastSegment.type, 1.0, status, y, map);
			else
				addFeature("E#MCASE_"+lastSegment.type, 1.0, status, y, map);
			
			if(lpattern.ctAlpha == 0 && lpattern.ctNum == 0)
				addFeature("E#NONUMALPHA", 1.0, status, y, map);
			else if(lpattern.ctAlpha != 0 && lpattern.ctNum == 0)
				addFeature("E#ONLYALPHA", 1.0, status, y, map);
			else if(lpattern.ctAlpha == 0 && lpattern.ctNum != 0)
				addFeature("E#ONLYNUM", 1.0, status, y, map);
			else
				addFeature("E#MIX", 1.0, status, y, map);
			
							
			if(tool.stopWord.contains(lastSegment.text)) // match the stop word
				addFeature("E#STOP_"+lastSegment.type, -1.0, status, y, map);
			else
				addFeature("E#STOP_"+lastSegment.type, 1.0, status, y, map);
			
			
			addFeature("E#WDLEN_"+lastSegment.type, lastSegment.text.length()/10.0, status, y, map);
			

			String bcHd = tool.entityBC.getPrefix(head.lemma());
			addFeature("E#HEADBC_"+lastSegment.type+"_"+bcHd, 1.0, status, y, map);
			
			// external
			if((tool.humando.contains(lastSegment.text) || tool.ctdmedic.contains(lastSegment.text)) 
					&& !tool.stopWord.contains(lastSegment.text)) {
				addFeature("E#DICTD_"+lastSegment.type, 1.0, status, y, map);
			}
			if((tool.chemElem.containsCaseSensitive(lastSegment.text) || tool.drugbank.contains(lastSegment.text) ||
					tool.jochem.contains(lastSegment.text) || tool.ctdchem.contains(lastSegment.text))
					&& !tool.stopWord.contains(lastSegment.text))
				addFeature("E#DICTC_"+lastSegment.type, 1.0, status, y, map);
			
			POS[] poses = {POS.NOUN, POS.ADJECTIVE};
			for(POS pos:poses) {
				ISynset synset = WordNetUtil.getMostSynset(tool.dict, hdLemmaLow, pos);
				if(synset!= null) {
					addFeature("E#HDSYNS"+lastSegment.type+"_"+synset.getID(),1.0, status, y, map);
				} 

				ISynset hypernym = WordNetUtil.getMostHypernym(tool.dict, hdLemmaLow, pos);
				if(hypernym!= null) {
					addFeature("E#HDHYPER"+lastSegment.type+"_"+hypernym.getID(),1.0, status, y, map);
				}
				
			}
			
			
		}
		
	}
	
	public static String segmentToPosSequence(Entity segment, PerceptronInputData1 input) {
		String posSeq = input.sentInfo.tokens.get(segment.start).tag();
		for(int j=segment.start+1;j<=segment.end;j++) {
			String currentPos = input.sentInfo.tokens.get(j).tag();
			posSeq += " "+currentPos;
		}
		return posSeq;
	}
	
}

