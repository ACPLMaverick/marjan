import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.sql.Date;
import com.thoughtworks.xstream.*;
import org.xmlpull.v1.XmlPullParser;

public class Serializacja {
	
	public void saveToXml(ArrayList<Eksponat> eksponaty, String path) 
	{
		XStream xstream = new XStream();
		String xml = xstream.toXML(eksponaty.get(0));
		File plik = new File(path);
		try
		{
			plik.createNewFile();
			FileWriter strumien = new FileWriter(plik);
			strumien.write(xml);
			strumien.close();
		}
		catch(IOException io)
		{
			System.out.println(io.getMessage());
		}
		catch(Exception e)
		{
			System.err.println("blad sec");
		}
	}
	
	public ArrayList<Eksponat> loadFromXml(String buffer)
	{
		XStream xstream = new XStream();
		xstream.alias("eksponat", Eksponat.class);
		xstream.alias("eksponaty", java.util.List.class);
		char buf[] = new char[1000];
		File plik = new File(buffer);
		try
		{
			FileReader strumien = new FileReader(buffer);
			strumien.read(buf);
		}
		catch(FileNotFoundException io)
		{
			System.out.println(io.getMessage());
		}
		catch(IOException io)
		{
			System.out.println(io.getMessage());
		}
		
		ArrayList<Eksponat> eksponaty = (ArrayList<Eksponat>) xstream.fromXML(buf.toString());
		return null;
	}
}
