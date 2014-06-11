package mainPackage;

import java.sql.*;
import com.sun.*;

/**
 * 
 * Klasa odpowiada za komunikacjê z baz¹ danych.
 *
 */
public class DBController {
	
	private Connection connection;
	
	public DBController()
	{
		this.connection = null;
	}
	
	public Repertoire getWholeRepertoire()
	{
		String command = "SELECT * FROM Films";

		try 
		{
			connect();
			
			Statement stat = connection.createStatement();
			ResultSet results = stat.executeQuery(command);
			
			stat.close();
			disconnect();
			
			while(results.next()) System.out.println(results.getObject("Title"));
		} 
		catch (ClassNotFoundException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		} 
		catch (SQLException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		
		return null;
	}
	
	private void connect() throws ClassNotFoundException, SQLException
	{
		String driver = "sun.jdbc.odbc.JdbcOdbcDriver";
		String connectionString = "jdbc:odbc:Driver= " +
								  "{Microsoft Access Driver (*.mdb, *.accdb)};DBQ=.\\Kino.accdb;DriverID=01";
		Class.forName(driver);
		this.connection = DriverManager.getConnection(connectionString);
	}
	
	private void disconnect() throws SQLException
	{
		connection.close();
		connection = null;
	}
}
