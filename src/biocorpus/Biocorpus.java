package biocorpus;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import cn.fox.nlp.SentenceSplitter;
import cn.fox.stanford.Tokenizer;
import cn.fox.utils.DTDEntityResolver;
import drug_side_effect_utils.Tool;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.PropertiesUtils;

public class Biocorpus {
	
	public static void buildBiocorpus() throws Exception {

		Tool tool = new Tool();
		tool.tokenizer = new Tokenizer(true, ' ');	
		tool.sentSplit = new SentenceSplitter(new Character[]{';'},false, "d:/dict/common_english_abbr.txt");
		OutputStreamWriter osw = new OutputStreamWriter(
				new FileOutputStream("F:/biocorpus/biocorpus.txt"), "utf-8");
		
		BufferedReader br = null;
		String thisLine = null;
		
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document d;
		
		
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream("F:/biomedical resource/ADE-Corpus-V2/ADE-NEG.txt"), "utf-8"));
		thisLine = null;
		while ((thisLine = br.readLine()) != null) {
			if(!thisLine.isEmpty()) {
				String str = thisLine.substring(thisLine.indexOf(" NEG ")+" NEG ".length());
				List<CoreLabel> tokens = tool.tokenizer.tokenize(0, str);
				for(CoreLabel token:tokens) {
					osw.write(token.word().toLowerCase()+" ");
				}
				osw.write("\n");
			} 
		}
		br.close();
		
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream("F:/biomedical resource/ADE-Corpus-V2/DRUG-AE.rel"), "utf-8"));
		thisLine = null;
		while ((thisLine = br.readLine()) != null) {
			if(!thisLine.isEmpty()) {
				String str1 = thisLine.substring(thisLine.indexOf("|")+1);
				
				String str = str1.substring(0, str1.indexOf("|"));
				List<CoreLabel> tokens = tool.tokenizer.tokenize(0, str);
				for(CoreLabel token:tokens) {
					osw.write(token.word().toLowerCase()+" ");
				}
				osw.write("\n");
			} 
		}
		br.close();
		
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream("F:/biomedical resource/Jochem/ChemlistV1_2.ontology"), "utf-8"));
		thisLine = null;
		while ((thisLine = br.readLine()) != null) {
			if(!thisLine.isEmpty()) {
				int i = thisLine.indexOf("DF ");
				if(-1 != i) {
					String str = thisLine.substring(i+"DF ".length());
					List<CoreLabel> tokens = tool.tokenizer.tokenize(0, str);
					for(CoreLabel token:tokens) {
						osw.write(token.word().toLowerCase()+" ");
					}
					osw.write("\n");
				}

			} 
		}
		br.close();
		
		
		
		
		db.setEntityResolver(new DTDEntityResolver("F:\\biomedical resource\\drugbank.xsd"));
		d = db.parse("F:\\biomedical resource\\drugbank.xml");
		NodeList drugBanks = d.getElementsByTagName("drugbank"); 
		NodeList drugNodes = ((Element)drugBanks.item(0)).getChildNodes();
		for(int i = 0; i < drugNodes.getLength(); i++) {
			Node node = drugNodes.item(i); 
			if(node.getParentNode().getNodeName().equals("drugbank") && node.getNodeType()==Node.ELEMENT_NODE) {
				Element drugNode = (Element)node;
				NodeList descriptionNodes = drugNode.getElementsByTagName("description");
				if(descriptionNodes.getLength()!=0) {
					Element e1 = (Element)descriptionNodes.item(0);
					if(e1.hasChildNodes()) {
						String temp = e1.getFirstChild().getNodeValue().trim();
						if(!temp.isEmpty()) {
							List<CoreLabel> tokens = tool.tokenizer.tokenize(0, temp);
							for(CoreLabel token:tokens) {
								osw.write(token.word().toLowerCase()+" ");
							}
							osw.write("\n");
						}
					}
				}
				
				NodeList indicationNodes = drugNode.getElementsByTagName("indication");
				if(indicationNodes.getLength()!=0) {
					Element e1 = (Element)indicationNodes.item(0);
					if(e1.hasChildNodes()) {
						String temp = e1.getFirstChild().getNodeValue().trim();
						if(!temp.isEmpty()) {
							List<CoreLabel> tokens = tool.tokenizer.tokenize(0, temp);
							for(CoreLabel token:tokens) {
								osw.write(token.word().toLowerCase()+" ");
							}
							osw.write("\n");
						}
					}
				}
				
				NodeList pharmacodynamicsNodes = drugNode.getElementsByTagName("pharmacodynamics");
				if(pharmacodynamicsNodes.getLength()!=0) {
					Element e1 = (Element)pharmacodynamicsNodes.item(0);
					if(e1.hasChildNodes()) {
						String temp = e1.getFirstChild().getNodeValue().trim();
						if(!temp.isEmpty()) {
							List<CoreLabel> tokens = tool.tokenizer.tokenize(0, temp);
							for(CoreLabel token:tokens) {
								osw.write(token.word().toLowerCase()+" ");
							}
							osw.write("\n");
						}
					}
				}
				
				NodeList mechanismNodes = drugNode.getElementsByTagName("mechanism-of-action");
				if(mechanismNodes.getLength()!=0) {
					Element e1 = (Element)mechanismNodes.item(0);
					if(e1.hasChildNodes()) {
						String temp = e1.getFirstChild().getNodeValue().trim();
						if(!temp.isEmpty()) {
							List<CoreLabel> tokens = tool.tokenizer.tokenize(0, temp);
							for(CoreLabel token:tokens) {
								osw.write(token.word().toLowerCase()+" ");
							}
							osw.write("\n");
						}
					}
				}
				
				NodeList toxicityNodes = drugNode.getElementsByTagName("toxicity");
				if(toxicityNodes.getLength()!=0) {
					Element e1 = (Element)toxicityNodes.item(0);
					if(e1.hasChildNodes()) {
						String temp = e1.getFirstChild().getNodeValue().trim();
						if(!temp.isEmpty()) {
							List<CoreLabel> tokens = tool.tokenizer.tokenize(0, temp);
							for(CoreLabel token:tokens) {
								osw.write(token.word().toLowerCase()+" ");
							}
							osw.write("\n");
						}
					}
				}
				
				NodeList metabolismNodes = drugNode.getElementsByTagName("metabolism");
				if(metabolismNodes.getLength()!=0) {
					Element e1 = (Element)metabolismNodes.item(0);
					if(e1.hasChildNodes()) {
						String temp = e1.getFirstChild().getNodeValue().trim();
						if(!temp.isEmpty()) {
							List<CoreLabel> tokens = tool.tokenizer.tokenize(0, temp);
							for(CoreLabel token:tokens) {
								osw.write(token.word().toLowerCase()+" ");
							}
							osw.write("\n");
						}
					}
				}
				
				NodeList absorptionNodes = drugNode.getElementsByTagName("absorption");
				if(absorptionNodes.getLength()!=0) {
					Element e1 = (Element)absorptionNodes.item(0);
					if(e1.hasChildNodes()) {
						String temp = e1.getFirstChild().getNodeValue().trim();
						if(!temp.isEmpty()) {
							List<CoreLabel> tokens = tool.tokenizer.tokenize(0, temp);
							for(CoreLabel token:tokens) {
								osw.write(token.word().toLowerCase()+" ");
							}
							osw.write("\n");
						}
					}
				}
					
			}
		}
		
		
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream("F:\\biomedical resource\\HumanDO.obo"), "utf-8"));
		thisLine = null;
		while ((thisLine = br.readLine()) != null) {
			if(!thisLine.isEmpty()) {
				int i = thisLine.indexOf("def: ");
				if(-1 != i) {
					String str = thisLine.substring(i+"def: ".length());
					List<CoreLabel> tokens = tool.tokenizer.tokenize(0, str);
					for(CoreLabel token:tokens) {
						osw.write(token.word().toLowerCase()+" ");
					}
					osw.write("\n");
				}

			} 
		}
		br.close();
		
		List<BufferedReader> brs= new ArrayList<>(Arrays.asList(
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biomedical resource/NCBI disease corpus/NCBItrainset_corpus.txt"), "utf-8")),
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biomedical resource/NCBI disease corpus/NCBItestset_corpus.txt"), "utf-8")),
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biomedical resource/NCBI disease corpus/NCBIdevelopset_corpus.txt"), "utf-8")),
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biocreative/v/cdr/CDR_Training_072115/CDR_TrainingSet.PubTator.PubTator.txt"), "utf-8")),
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biocreative/v/cdr/CDR_Dev_072115/CDR_DevelopmentSet.PubTator.txt"), "utf-8"))
				));
		for(int k=0;k<brs.size();k++) {
			br = brs.get(k);
			thisLine = null;
			while ((thisLine = br.readLine()) != null) {
				if(!thisLine.isEmpty()) {
					int i = thisLine.indexOf("|t|");
					if(-1 != i) {
						String str = thisLine.substring(i+"|t|".length());
						List<CoreLabel> tokens = tool.tokenizer.tokenize(0, str);
						for(CoreLabel token:tokens) {
							osw.write(token.word().toLowerCase()+" ");
						}
						osw.write("\n");
					}
					i = thisLine.indexOf("|a|");
					if(-1 != i) {
						String str = thisLine.substring(i+"|a|".length());
						List<CoreLabel> tokens = tool.tokenizer.tokenize(0, str);
						for(CoreLabel token:tokens) {
							osw.write(token.word().toLowerCase()+" ");
						}
						osw.write("\n");
					}

				} 
			}
			br.close();
		}
		
		
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream("F:/biocreative/v/bel/training_data/Training.sentence"), "utf-8"));
		thisLine = null;
		while ((thisLine = br.readLine()) != null) {
			if(!thisLine.isEmpty()) {
				String[] splitted = thisLine.split("\t");
				List<CoreLabel> tokens = tool.tokenizer.tokenize(0, splitted[2]);
				for(CoreLabel token:tokens) {
					osw.write(token.word().toLowerCase()+" ");
				}
				osw.write("\n");

			} 
		}
		br.close();
		
		brs= new ArrayList<>(Arrays.asList(
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biocreative/iv/chemdner/train_development_abstract/chemdner_abs_training.txt"), "utf-8")),
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biocreative/iv/chemdner/train_development_abstract/chemdner_abs_development.txt"), "utf-8")),
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biocreative/iv/chemdner/CHEMDNER_TEST_V01/chemdner_abs_test.txt"), "utf-8"))
				));
		for(int k=0;k<brs.size();k++) {
			br = brs.get(k);
			thisLine = null;
			while ((thisLine = br.readLine()) != null) {
				if(!thisLine.isEmpty()) {
					String[] splitted = thisLine.split("\t");
					List<CoreLabel> tokens = tool.tokenizer.tokenize(0, splitted[1]);
					for(CoreLabel token:tokens) {
						osw.write(token.word().toLowerCase()+" ");
					}
					osw.write("\n");
					
					tokens = tool.tokenizer.tokenize(0, splitted[2]);
					for(CoreLabel token:tokens) {
						osw.write(token.word().toLowerCase()+" ");
					}
					osw.write("\n");
	
				} 
			}
			br.close();
		}
		
		
		brs= new ArrayList<>(Arrays.asList(
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biocreative/v/chemdner/cemp_training_set/chemdner_patents_train_text.txt"), "utf-8")),
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biocreative/v/chemdner/cemp_development_set_v03/chemdner_patents_development_text.txt"), "utf-8")),
				new BufferedReader(new InputStreamReader(
						new FileInputStream("F:/biocreative/v/chemdner/CHEMDNER_TEST_TEXT/chemdner_patents_test_background_text_release.txt"), "utf-8"))
				));
		for(int k=0;k<brs.size();k++) {
			br = brs.get(k);
			thisLine = null;
			while ((thisLine = br.readLine()) != null) {
				if(!thisLine.isEmpty()) {
					String[] splitted = thisLine.split("\t");
					List<CoreLabel> tokens = tool.tokenizer.tokenize(0, splitted[1]);
					for(CoreLabel token:tokens) {
						osw.write(token.word().toLowerCase()+" ");
					}
					osw.write("\n");
					
					tokens = tool.tokenizer.tokenize(0, splitted[2]);
					for(CoreLabel token:tokens) {
						osw.write(token.word().toLowerCase()+" ");
					}
					osw.write("\n");
	
				} 
			}
			br.close();
		}
		
		
		
		
		
		osw.close();
	
	}
	
	public static void merge() throws Exception {
		OutputStreamWriter osw = new OutputStreamWriter(
				new FileOutputStream("F:/biocorpus/biocorpus_bionlplab.txt"), "utf-8");
		
		BufferedReader br = null;
		String thisLine = null;
		
				
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream("F:/biocorpus/biocorpus.txt"), "utf-8"));
		thisLine = null;
		while ((thisLine = br.readLine()) != null) {
			if(!thisLine.isEmpty()) {
				osw.write(thisLine+"\n");
			} 
		}
		br.close();
		
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream("F:/biomedical resource/bionlplaborg/1-grams-decode.txt"), "utf-8"));
		thisLine = null;
		while ((thisLine = br.readLine()) != null) {
			if(!thisLine.isEmpty()) {
				osw.write(thisLine+"\n");
			} 
		}
		br.close();
		
		
		osw.close();
	}

	public static void main(String[] args) throws Exception{
		
		merge();
	}
	
	

}
