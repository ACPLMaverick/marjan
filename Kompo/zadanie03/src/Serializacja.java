import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.sql.Date;
import com.thoughtworks.xstream.*;
import org.xmlpull.v1.XmlPullParser;

class Serializacja {
	
	private ArrayList<Eksponat> mojeEksponaty;
	private String lokalizacjaPliku;
	private Boolean isSaved;
	
	public Serializacja(ArrayList<Eksponat> eksponaty, String lokalizacjaPliku)
	{
		this.mojeEksponaty = eksponaty;
		this.lokalizacjaPliku = lokalizacjaPliku;
	}
	
	public void saveToXml() 
	{
		XStream xstream = new XStream();
		String xml = xstream.toXML(mojeEksponaty);
		File plik = new File(lokalizacjaPliku);
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
		isSaved = true;
	}
	
	@SuppressWarnings("unchecked")
	public ArrayList<Eksponat> loadFromXml()
	{
		if(isSaved)
		{
			XStream xstream = new XStream();
			File plik = new File(lokalizacjaPliku);
			mojeEksponaty = (ArrayList<Eksponat>)xstream.fromXML(plik);
			return mojeEksponaty;
		}
		else
		{
			return null;
		}
	}
	public void wyswietl()
	{
		if(isSaved)
		{
			for(int i = 0; i<mojeEksponaty.size(); i++)
			{
				System.out.println(mojeEksponaty.get(i).toString() + "\n");
			}
		}
	}
}
