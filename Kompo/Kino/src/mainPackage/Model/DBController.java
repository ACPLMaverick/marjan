package mainPackage.Model;

import java.util.*;
import java.util.Date;
import java.sql.*;
import java.util.ArrayList;

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
	
	/**
	 * Metoda s³u¿y do pobrania ca³ego repertuaru z bazy danych. Zwraca obiekt typu Repertoire
	 *
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
	 * Metoda s³u¿y do dodawania nowego seansu do bazy danych.
	 *
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
	 * Metoda s³u¿y do aktualizacji istniej¹cego seansu w bazie danych. Wyszukiwanie po dacie
	 *
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
	 * Metoda s³u¿y do usuwania seansu z bazy danych. Wyszukiwanie po dacie.
	 *
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
	 * Metoda s³u¿y do dodawania nowego filmu do bazy danych.
	 * @param filmData dane tablicy musi odpowiadaæ kolumnom tabeli: Title, Genre, TicketPrice, LicensePrice
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
	 * Metoda s³u¿y do aktualizacji istniej¹cego filmu w bazie danych.
	 * @param filmData dane tablicy musi odpowiadaæ kolumnom tabeli: Title, Genre, TicketPrice, LicensePrice
	 * @param filmName po nim wyszukujemy zadany film do modyfikacji
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
	 * Metoda s³u¿y do usuwania filmu z bazy danych.
	 * @param filmName po nim wyszukujemy zadany film do usuniêcia
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
	 * Metoda s³u¿y do pobierania filmu z bazy danych.
	 * @param filmName po nim wyszukujemy zadany film do usuniêcia
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
	 * Metoda s³u¿y do pobrania wszystkich filmów z bazy danych
	 * 
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
