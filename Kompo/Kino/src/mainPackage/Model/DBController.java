package mainPackage.Model;

import java.util.*;
import java.util.Date;
import java.sql.*;
import java.util.ArrayList;

import com.sun.*;

// TODO: Auto-generated Javadoc
/**
 * 
 * Klasa odpowiada za komunikacje z baza danych.
 *
 */
public class DBController {

	private Connection connection;
	
	/**
	 * Tworzy nowy obiekt typu DBController.
	 */
	public DBController()
	{
		this.connection = null;
	}
	
	/**
	 * Pobiera caly repertuar z bazy danych.
	 *
	 * @return caly repertuar.
	 */
	public ArrayList<Seance> getWholeRepertoire()
	{
		String command = "SELECT Films.Title, Films.Genre, Seances.SeanceDate, Films.TicketPrice, Films.LicensePrice "
						+ "FROM Seances INNER JOIN Films ON Seances.ID_film = Films.ID";
		ArrayList<String> resultString = new ArrayList<String>();

		try 
		{
			connect();
			
			Statement stat = connection.createStatement();
			ResultSet results = stat.executeQuery(command);
			
			while(results.next())
			{
				String myString = "";
				for(int i = 1; i <= 5; i++)
				{
					myString += (results.getString(i) + ";"); 
				}
				resultString.add(myString);	
			}
			
			stat.close();
			disconnect();
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
		
		ArrayList<Seance> newRep = new ArrayList<Seance>();
		for(String str : resultString)
		{
			String[] elements = str.split(";");
			
			String dateString = elements[2];
			String[] dateStringElements = dateString.split(" ");
			String[] dateElements = dateStringElements[0].split("-");
			String[] hourElements = dateStringElements[1].split(":");
			GregorianCalendar cal = new GregorianCalendar(Integer.valueOf(dateElements[0]), Integer.valueOf(dateElements[1]) - 1, Integer.valueOf(dateElements[2]), Integer.valueOf(hourElements[0]), Integer.valueOf(hourElements[1]));
			Date date = cal.getTime();
			Seance newSeance = new Seance(new Film(elements[0], elements[1], Double.valueOf(elements[3]), Double.valueOf(elements[4])), date, 0);
			//Seance newSeance = new Seance(elements[1], elements[0], date, 0, Double.valueOf(elements[3]));
			newRep.add(newSeance);
		}
		
		return newRep;
	}
	
	/**
	 * Dodaje nowy seans do bazy danych.
	 *
	 * @param seance seans.
	 */
	public void addSeance(Seance seance)
	{
		String commandGetID = "SELECT Films.ID FROM Films WHERE Films.Title = '" + seance.getTitle() + "'";
		try 
		{
			connect();
			Statement stat = connection.createStatement();
			ResultSet rs = stat.executeQuery(commandGetID);
			rs.next();
			String id = rs.getString(1);
			String command = 
					"INSERT INTO Seances (ID_film, SeanceDate) "
					+ "VALUES (" + id +", '" + seance.getDateAsString() + "')";
			stat.executeQuery(command);
			stat.close();
			disconnect();
		} 
		catch (ClassNotFoundException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		catch (SQLException e) 
		{
			// TODO Auto-generated catch block
			if(e.getMessage() != "No ResultSet was produced")e.printStackTrace();
		}
	}
	
	/**
	 * Aktualizuje istniejący seans w bazie danych. Wyszukiwanie po dacie.
	 *
	 * @param seance seans do zaktualizowania.
	 * @param dateTime data (kryterium wyszukiwania).
	 */
	public void updateSeance(Seance seance, String dateTime)
	{
		String commandGetID = "SELECT Films.ID FROM Films WHERE Films.Title = '" + seance.getTitle() + "'";
		try 
		{
			connect();
			Statement stat = connection.createStatement();
			ResultSet rs = stat.executeQuery(commandGetID);
			rs.next();
			String id = rs.getString(1);
			String command = 
					"UPDATE Seances "
					+ "SET ID_film=" + id + ", SeanceDate='" + seance.getDateAsString() + "' "
					+ "WHERE SeanceDate=#" + dateTime + "#";
			stat.executeQuery(command);
			stat.close();
			disconnect();
		} 
		catch (ClassNotFoundException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		catch (SQLException e) 
		{
			// TODO Auto-generated catch block
			if(e.getMessage() != "No ResultSet was produced")e.printStackTrace();
		}
	}
	
	/**
	 * Usuwa seans z bazy danych. Wyszukiwanie po dacie.
	 *
	 * @param dateTime data (kryterium wyszukiwania).
	 */
	public void deleteSeance(String dateTime)
	{
		String command = 
				"DELETE FROM Seances "
				+ "WHERE SeanceDate=#" + dateTime + "# ";
				
		try 
		{
			connect();
			Statement stat = connection.createStatement();
			stat.executeQuery(command);
			stat.close();
			disconnect();
		} 
		catch (ClassNotFoundException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		catch (SQLException e) 
		{
			// TODO Auto-generated catch block
			if(e.getMessage() != "No ResultSet was produced")e.printStackTrace();
		}
	}
	
	/**
	 * Dodaje nowy film do bazy danych.
	 *
	 * @param film film.
	 */
	public void addFilm(Film film)
	{
		String command = 
				"INSERT INTO Films (Title, Genre, TicketPrice, LicensePrice) "
						+ "VALUES ('" + film.getTitle() +"', '" + film.getGenre() + "', " + film.getPriceAsString() + ", " + film.getLicensePriceAsString() + ")";
				
		try 
		{
			connect();
			Statement stat = connection.createStatement();
			stat.executeQuery(command);
			stat.close();
			disconnect();
		} 
		catch (ClassNotFoundException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		catch (SQLException e) 
		{
			// TODO Auto-generated catch block
			if(e.getMessage() != "No ResultSet was produced")e.printStackTrace();
		}
	}
	
	/**
	 * Aktualizuje istniejący film w bazie danych. Wyszukiwanie po tytule filmu.
	 *
	 * @param film film.
	 * @param filmName tytul filmu (kryterium wyszukiwania).
	 */
	public void updateFilm(Film film, String filmName)
	{
		String command = 
				"UPDATE Films "
				+ "SET Title='" + film.getTitle() +"', Genre='" + film.getGenre() + "', TicketPrice=" + film.getPriceAsString() + ", LicensePrice=" + film.getLicensePriceAsString() + " "
				+ "WHERE Title='" + filmName + "'";
				
		try 
		{
			connect();
			Statement stat = connection.createStatement();
			stat.executeQuery(command);
			stat.close();
			disconnect();
		} 
		catch (ClassNotFoundException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		catch (SQLException e) 
		{
			// TODO Auto-generated catch block
			if(e.getMessage() != "No ResultSet was produced")e.printStackTrace();
		}
	}
	
	/**
	 * Usuwa film z bazy danych. Wyszukiwanie po tytule filmu.
	 * 
	 * @param filmName tytul filmu (kryterium wyszukiwania).
	 */
	public void deleteFilm(String filmName)
	{
		String command = 
				"DELETE FROM Films "
				+ "WHERE Title='" + filmName + "' ";
				
		try 
		{
			connect();
			Statement stat = connection.createStatement();
			stat.executeQuery(command);
			stat.close();
			disconnect();
		} 
		catch (ClassNotFoundException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		catch (SQLException e) 
		{
			// TODO Auto-generated catch block
			if(e.getMessage() != "No ResultSet was produced")e.printStackTrace();
		}
	}
	
	/**
	 * Pobiera film z bazy danych o konkretnym tytule.
	 *
	 * @param filmName tytul filmu (kryterium wyszukiwania).
	 * @return Film o konkretnym tytule.
	 */
	public Film getFilm(String filmName)
	{
		String command = "SELECT TOP 1 Films.Title, Films.Genre, Films.TicketPrice, Films.LicensePrice "
				+ "FROM Films "
				+ "WHERE Title = '" + filmName + "' ";
		ArrayList<String> resultString = new ArrayList<String>();
		
		try 
		{
			connect();
			
			Statement stat = connection.createStatement();
			ResultSet results = stat.executeQuery(command);
			
			while(results.next())
			{
				for(int i = 1; i <= 4; i++)
				{
					resultString.add(results.getString(i));
				}	
			}
			
			stat.close();
			disconnect();
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
		return new Film(resultString.get(0), resultString.get(1), Double.valueOf(resultString.get(2)), Double.valueOf(resultString.get(3)));
	}
	
	/**
	 * Pobiera wszystkie filmy z bazy danych.
	 *
	 * @return Kolekcje filmow.
	 */
	public ArrayList<Film> getAllFilms()
	{
		String command = "SELECT Films.Title, Films.Genre, Films.TicketPrice, Films.LicensePrice "
				+ "FROM Films";
		ArrayList<Film> resultString = new ArrayList<Film>();
		
		try 
		{
			connect();
			
			Statement stat = connection.createStatement();
			ResultSet results = stat.executeQuery(command);
			
			while(results.next())
			{
				ArrayList<String> myString = new ArrayList<String>();
				for(int i = 1; i <= 4; i++)
				{
					myString.add(results.getString(i));
				}
				resultString.add(new Film(myString.get(0), myString.get(1), Double.valueOf(myString.get(2)), Double.valueOf(myString.get(3))));	
			}
			
			stat.close();
			disconnect();
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
		return resultString;
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
