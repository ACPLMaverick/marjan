package mainPackage.Model;

import java.io.*;

import com.thoughtworks.xstream.XStream;

public class SerializationController<T> {
	
	private T myCollection;
	
	public SerializationController(T myCollection)
	{
		this.myCollection = myCollection;
	}
	
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
