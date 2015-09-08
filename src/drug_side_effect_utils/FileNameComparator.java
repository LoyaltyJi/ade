package drug_side_effect_utils;

import java.io.File;
import java.util.Comparator;

public class FileNameComparator implements Comparator<File> {

	@Override
	public int compare(File o1, File o2) {
		/*String file1 = o1.getName();
    	long docID1 = Long.parseLong(file1.substring(0, file1.indexOf("-")));
        String file2 = o2.getName();
        long docID2 = Long.parseLong(file2.substring(0, file2.indexOf("-")));
        if(docID1>docID2) return 1;
        else if(docID1<docID2) return -1;
        else {
        	long sentID1 = Long.parseLong(file1.substring(file1.indexOf("-")+1, file1.lastIndexOf(".")));
        	long sentID2 = Long.parseLong(file2.substring(file2.indexOf("-")+1, file2.lastIndexOf(".")));
        	if(sentID1>sentID2) return 1;
	        else if(sentID1<sentID2) return -1;
	        else return 0;
        }*/
		String file1 = o1.getName();
        String file2 = o2.getName();
		long sentID1 = Long.parseLong(file1.substring(0, file1.indexOf(".")));
    	long sentID2 = Long.parseLong(file2.substring(0, file2.indexOf(".")));
    	if(sentID1>sentID2) return 1;
        else if(sentID1<sentID2) return -1;
        else return 0;
	}

	

}
