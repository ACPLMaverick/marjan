/*
 * 
 */
package mainPackage.Controller;

import java.io.*;

import com.thoughtworks.xstream.XStream;

// TODO: Auto-generated Javadoc
/**
 * Klasa generyczna uzywana do serializacji i deserializacji kolekcji w formacie XML
 */
public class SerializationController<T> {
	
	private T myCollection;
	
	/**
	 * Tworzy nowy obiekt klasy SerializationController.
	 *
	 * @param myCollection kolekcja elementow typu generycznego.
	 */
	public SerializationController(T myCollection)
	{
		this.myCollection = myCollection;
	}
	
	/**
	 * Serializuje kolekcje do pliku XML o podanej sciezce.
	 *
	 * @param path sciezka okreslajaca, gdzie ma zostac stworzony plik XML.
	 */
	public void serialize(String path)
	{
		if(myCollection == null) throw new NullPointerException("Collection not initialised!");
		else
		{
			XStream xstream = new XStream();
			String xml = xstream.toXML(myCollection);
			
			File myFile = new File(path);
			
			try
			{
				myFile.createNewFile();
				FileWriter stream = new FileWriter(myFile);
				stream.write(xml);
				stream.close();
			}
			catch(IOException e)
			{
				System.out.println(e.getMessage());
			}
		}
	}
	
	/**
	 * Deserializuje plik XML o podanej sciezce
	 *
	 * @param path sciezka do pliku XML.
	 * @return Kolekcje ktora zawiera zdeserializowane dane.
	 */
	public T deserialize(String path)
	{

		XStream xstream = new XStream();
		File myFile = new File(path);
		try
		{
			this.myCollection = (T)xstream.fromXML(myFile);
		}
		catch(Exception e)
		{
			e.printStackTrace();
			return null;
		}
		return this.myCollection;
	}
}
