package joint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import cn.fox.machine_learning.Perceptron;
import cn.fox.machine_learning.PerceptronFeatureFunction;
import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.machine_learning.PerceptronStatus;
import cn.fox.nlp.EnglishPos;
import cn.fox.nlp.Punctuation;
import cn.fox.stanford.StanfordTree;
import cn.fox.utils.WordNetUtil;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;
import drug_side_effect_utils.Tool;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.Tree;
import gnu.trove.TObjectDoubleHashMap;



public class RelationFeatures extends PerceptronFeatureFunction {
	public RelationFeatures(Perceptron perceptron) {
		super(perceptron);
	}
	
	@Override
	public void compute(PerceptronInputData x, PerceptronStatus status,
			PerceptronOutputData y, Object other,
			TObjectDoubleHashMap<String> map) {
		PerceptronInputData1 input = (PerceptronInputData1)x;
		PerceptronOutputData1 output = (PerceptronOutputData1)y;
		Tool tool = (Tool)other;
		
		int lastSegmentIndex = output.getLastSegmentIndex(status.tokenIndex);
		Entity latter = output.segments.get(lastSegmentIndex);
		
		if(latter.type.equals(Perceptron.EMPTY))
			return;
		
		int countEntity = 0;
		
		for(int index=lastSegmentIndex-1;index>=0;index--) {
			Entity former = output.segments.get(index);
			if(former.type.equals(Perceptron.EMPTY))
				continue;
			else {
				countEntity++;
			}
			
			String type = Perceptron.EMPTY;
			if(output.relations.contains(new RelationEntity("CID", former, latter)))
				type = "CID";
			
			CoreLabel clHdFormer = AdeJointMain.getHeadWordOfSegment(former, input);
			CoreLabel clHdLatter = AdeJointMain.getHeadWordOfSegment(latter, input);
			String hdFormer = clHdFormer.lemma().toLowerCase();
			String hdLatter = clHdLatter.lemma().toLowerCase();
			{
				// lexical
				addFeature("#EN_"+type+"_"+former.text.toLowerCase()+"_"+latter.text.toLowerCase(), 1.0, status, y, map);
				addFeature("#EN_"+type+"_"+latter.text.toLowerCase()+"_"+former.text.toLowerCase(), 1.0, status, y, map);
				
				
				
				addFeature("#HM_"+hdFormer+"_"+hdLatter+"_"+type, 1.0, status, y, map);
				addFeature("#HM_"+hdLatter+"_"+hdFormer+"_"+type, 1.0, status, y, map);
				
				addFeature("#HMPOS_"+clHdFormer.tag()+"_"+clHdLatter.tag()+"_"+type, 1.0, status, y, map);
				addFeature("#HMPOS_"+clHdLatter.tag()+"_"+clHdFormer.tag()+"_"+type, 1.0, status, y, map);
				
				int len1 = hdFormer.length()>4 ? 4:hdFormer.length();
				int len2 = hdLatter.length()>4 ? 4:hdLatter.length();
				String hd1Pref = hdFormer.substring(0, len1);
				String hd1Suf = hdFormer.substring(hdFormer.length()-len1, hdFormer.length());
				String hd2Pref = hdLatter.substring(0, len2);
				String hd2Suf = hdLatter.substring(hdLatter.length()-len2, hdLatter.length());
				addFeature("#HD12PREF_"+hd1Pref+"_"+hd2Pref+"_"+type,1.0, status, y, map);
				addFeature("#HD12PREF_"+hd2Pref+"_"+hd1Pref+"_"+type,1.0, status, y, map);
				addFeature("#HD12SUF_"+hd1Suf+"_"+hd2Suf+"_"+type,1.0, status, y, map);
				addFeature("#HD12SUF_"+hd2Suf+"_"+hd1Suf+"_"+type,1.0, status, y, map);
				addFeature("#HD12PREFSUF_"+hd1Pref+"_"+hd1Suf+"_"+hd2Pref+"_"+hd2Suf+"_"+type,1.0, status, y, map);
				addFeature("#HD12PREFSUF_"+hd2Pref+"_"+hd2Suf+"_"+hd1Pref+"_"+hd1Suf+"_"+type,1.0, status, y, map);
				
				
				// context
				if(former.end+1==latter.start)
					addFeature("#WBNULL"+"_"+type, 1.0, status, y, map);
				else if(latter.start-former.end==2)
					addFeature("#WBFL"+input.sentInfo.tokens.get(former.end+1).lemma().toLowerCase()+"_"+type, 1.0, status, y, map);
				else {
					for(int i=former.end+1;i<latter.start;i++) {
						EnglishPos.Type posType = EnglishPos.getType(input.sentInfo.tokens.get(i).tag());
						if(posType==EnglishPos.Type.NOUN || posType==EnglishPos.Type.VERB || posType==EnglishPos.Type.ADJ ||
								posType==EnglishPos.Type.PREP)
							map.put("#WB_"+input.sentInfo.tokens.get(i).lemma().toLowerCase()+"_"+type,1.0); 
					}
				}
				
				for(int i=former.start-1, count=2;count>0 && i>=0; count--,i--) {
					EnglishPos.Type posType = EnglishPos.getType(input.sentInfo.tokens.get(i).tag());
					if(posType==EnglishPos.Type.NOUN || posType==EnglishPos.Type.VERB || posType==EnglishPos.Type.ADJ ||
							posType==EnglishPos.Type.PREP)
						map.put("#BM1_"+input.sentInfo.tokens.get(i).lemma().toLowerCase()+"_"+type,1.0); 
				}
				for(int i=latter.end+1, count=2;count>0 && i<input.sentInfo.tokens.size(); count--,i++) {
					EnglishPos.Type posType = EnglishPos.getType(input.sentInfo.tokens.get(i).tag());
					if(posType==EnglishPos.Type.NOUN || posType==EnglishPos.Type.VERB || posType==EnglishPos.Type.ADJ ||
							posType==EnglishPos.Type.PREP)
						map.put("#AM2_"+input.sentInfo.tokens.get(i).lemma().toLowerCase()+"_"+type,1.0); 
				}
				
				addFeature("#WDBETNUM_"+type , (latter.start-former.end)*1.0/10, status, y, map);
				addFeature("#ENBETNUM_"+type,(countEntity-1)*1.0/10, status, y, map);
				

				
				
				
				// type
				addFeature("#ET_"+type+"_"+former.type+"_"+latter.type, 1.0, status, y, map);
				addFeature("#ET_"+type+"_"+latter.type+"_"+former.type, 1.0, status, y, map);
				
				String bcHdFormer = tool.entityBC.getPrefix(hdFormer);
				String bcHdLatter = tool.entityBC.getPrefix(hdLatter);
				addFeature("#HMBC_"+bcHdFormer+"_"+bcHdLatter+"_"+type, 1.0, status, y, map);
				addFeature("#HMBC_"+bcHdLatter+"_"+bcHdFormer+"_"+type, 1.0, status, y, map);

			}
			
			// external
			if(former.type.equals("Chemical") && latter.type.equals("Disease")) {
				if(tool.sider.contains(former.text, latter.text)) {
					addFeature("#DICTSD_"+type,1.0, status, y, map);
				}
			}
			else if(former.type.equals("Disease") && latter.type.equals("Chemical")) {
				if(tool.sider.contains(latter.text, former.text)) {
					addFeature("#DICTSD_"+type,1.0, status, y, map);
				}
			} 
			
			{
				POS[] poses = {POS.NOUN, POS.ADJECTIVE};
				for(POS pos:poses) {
					ISynset synset = WordNetUtil.getMostSynset(tool.dict, hdFormer, pos);
					if(synset!= null) {
						map.put("#HD1SYNS_"+synset.getID()+"_"+type,1.0);
					} 

					ISynset hypernym = WordNetUtil.getMostHypernym(tool.dict, hdFormer, pos);
					if(hypernym!= null) {
						map.put("#HD1HYPER_"+hypernym.getID()+"_"+type,1.0);
					}
					
				}
			}
			
			{
				POS[] poses = {POS.NOUN, POS.ADJECTIVE};
				for(POS pos:poses) {
					ISynset synset = WordNetUtil.getMostSynset(tool.dict, hdLatter, pos);
					if(synset!= null) {
						map.put("#HD2SYNS_"+synset.getID()+"_"+type,1.0);
					} 

					ISynset hypernym = WordNetUtil.getMostHypernym(tool.dict, hdLatter, pos);
					if(hypernym!= null) {
						map.put("#HD2HYPER_"+hypernym.getID()+"_"+type,1.0);
					}
					
				}
			}
							
			if(input.sentInfo.root!=null) {
				Tree nodeFormer = getSegmentTreeNode(former, input);
				Tree nodeLatter = getSegmentTreeNode(latter, input);
				
				Tree common = StanfordTree.getCommonAncestor(input.sentInfo.root, nodeFormer, nodeLatter);
				addFeature("#PAR_"+common.value()+"_"+type, 1.0, status, y, map);
				List<Tree> path = input.sentInfo.root.pathNodeToNode(nodeFormer, nodeLatter);
				ArrayList<Tree> phrasePathDeleteOverlap = new ArrayList<Tree>();
				String lastNodeValue = "";
				String featurePath =  "#CPP_";
				for(int k=0;k<path.size();k++) {
					Tree node  = path.get(k);
					if(node.isPhrasal()) {
						if(!node.value().equals(lastNodeValue)) {
							phrasePathDeleteOverlap.add(node);
							featurePath += node.value();
						}
						lastNodeValue = node.value();
					}
				}
				if(phrasePathDeleteOverlap.size()==0)
					addFeature("#CPHBNULL"+"_"+type, 1.0, status, y, map);
				else if(phrasePathDeleteOverlap.size()==1)
					addFeature("#CPHBFL_"+lastNodeValue+"_"+type, 1.0, status, y, map);
				else {
					addFeature("#CPHBF_"+phrasePathDeleteOverlap.get(0).value()+"_"+type, 1.0, status, y, map);
					addFeature("#CPHBL_"+phrasePathDeleteOverlap.get(phrasePathDeleteOverlap.size()-1).value()+"_"+type, 1.0, status, y, map);
					addFeature(featurePath+"_"+type, 1.0, status, y, map);
				}
				
			}
			
			
			if(input.sentInfo.depGraph!=null) {
				// there may be someone who isn't in the semantic graph
				IndexedWord nodeFormer  = input.sentInfo.depGraph.getNodeByIndexSafe(clHdFormer.index());
				if(nodeFormer != null) {
					List<SemanticGraphEdge> edges = input.sentInfo.depGraph.incomingEdgeList(nodeFormer);
					for(int i=0;i<edges.size();i++) {
						CoreLabel clGw  = edges.get(i).getGovernor();
						String gw = clGw.lemma().toLowerCase();
						addFeature("#EN1GW_"+gw+"_"+type, 1.0, status, y, map);
						addFeature("#EN1GWPOS_"+clGw.tag()+"_"+type, 1.0, status, y, map);
						addFeature("#EN1TP_GW_"+former.type+"_"+gw+"_"+type, 1.0, status, y, map);
						addFeature("#HD1_GW_"+hdFormer+"_"+gw+"_"+type, 1.0, status, y, map);

						break; // only take one dependent
					}
				}
				
				IndexedWord nodeLatter  = input.sentInfo.depGraph.getNodeByIndexSafe(clHdLatter.index());
				if(nodeLatter != null) {
					List<SemanticGraphEdge> edges = input.sentInfo.depGraph.incomingEdgeList(nodeLatter);
					for(int i=0;i<edges.size();i++) {
						CoreLabel clGw  = edges.get(i).getGovernor();
						String gw = clGw.lemma().toLowerCase();
						addFeature("#EN2GW_"+gw+"_"+type, 1.0, status, y, map);
						addFeature("#EN2GWPOS_"+clGw.tag()+"_"+type, 1.0, status, y, map);
						addFeature("#EN2TP_GW_"+latter.type+"_"+gw+"_"+type, 1.0, status, y, map);
						addFeature("#HD2_GW_"+hdLatter+"_"+gw+"_"+type, 1.0, status, y, map);
						
						break; // only take one dependent
					}
				}
			}
			
			
			
			
		}
			
			
			
		
	}
	
	public static Tree getSegmentTreeNode(Entity segment, PerceptronInputData1 input) {
		
		List<Tree> nodes = input.sentInfo.root.getLeaves();
		if(segment.start==segment.end)  // return the leaf node
			return nodes.get(input.sentInfo.tokens.get(segment.start).index()-1);
		else { // return the phrasal node
			Tree nodeFormer = nodes.get(input.sentInfo.tokens.get(segment.start).index()-1);
			Tree nodeLatter = nodes.get(input.sentInfo.tokens.get(segment.end).index()-1);
			Tree common = StanfordTree.getCommonAncestor(input.sentInfo.root, nodeFormer, nodeLatter);
			return common; 
		}
			
	}
	

}
