package mainPackage.Model;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;

// TODO: Auto-generated Javadoc
/**
 * Reprezentuje jeden, konkretny seans filmowy.
 */
public class Seance {
	
	private Film film;
	private Date date;
	private int seatPlan;
	
	/** Stala okreslajaca ilosc pol jakie opisuja seans w interfejsie. */
	public static final int fieldsCount = 5;
	
	/**
	 * Tworzy nowy obiekt typu Seance.
	 */
	public Seance()
	{
		this.film = new Film();
		this.date = new Date();
		this.seatPlan = 0;
	}
	
	/**
	 * Tworzy nowy obiekt typu Seance z konkretnymi parametrami.
	 *
	 * @param film wyswietlany film.
	 * @param date data seansu.
	 * @param seatPlan ilosc wolnych miejsc.
	 */
	public Seance(Film film, Date date, int seatPlan)
	{
		this.film = film;
		this.date = date;
		this.seatPlan = seatPlan;
	}
	
	/**
	 * Zwraca wyswietlany film.
	 *
	 * @return Wyswietlany film.
	 */
	public Film getFilm() { return film; }
	
	/**
	 * Zwraca gatunek wyswietlanego filmu.
	 *
	 * @return Gatunek wyswietlanego filmu.
	 */
	public String getGenre() { return this.film.getGenre(); }
	
	/**
	 * Zwraca tytul wyswietlanego filmu.
	 *
	 * @return Tytul wyswietlanego filmu.
	 */
	public String getTitle() { return this.film.getTitle(); }
	
	/**
	 * Zwraca date seansu.
	 *
	 * @return Date seansu.
	 */
	public Date getDate() { return date; }
	
	/**
	 * Zwraca ilosc wolnych miejsc na dany seans.
	 *
	 * @return Ilosc wolnych miejsc.
	 */
	public int getSeatPlan() { return seatPlan; }
	
	/**
	 * Ustawia ilosc wolnych miejsc na danyc seans.
	 *
	 * @param i nowa ilosc wolnych miejsc.
	 */
	public void setSeatPlan(int i) { seatPlan = i; };
	
	/**
	 * Zwraca cene za bilet.
	 *
	 * @return Cene.
	 */
	public double getPrice() { return this.film.getPrice(); }
	
	/**
	 * Zwraca date seansu jako String.
	 *
	 * @return Date seansu jako String.
	 */
	public String getDateAsString() 
	{ 
		/*
		Calendar cal = Calendar.getInstance();
		cal.setTime(date);
		String myTime = String.valueOf(cal.get(Calendar.HOUR)) + ":" + String.valueOf(cal.get(Calendar.MINUTE));
		String myDate = String.valueOf(cal.get(Calendar.DAY_OF_MONTH)) + "-" + String.valueOf(cal.get(Calendar.MONTH)) + "-" + String.valueOf(cal.get(Calendar.YEAR));
		String finalDate = myDate + " " + myTime;
		return finalDate; 
		*/
		DateFormat df = new SimpleDateFormat("dd-MM-yyyy HH:mm");
		return df.format(this.date);
	}
	
	/**
	 * Zwraca cene biletu jako String.
	 *
	 * @return Cene biletu jako String.
	 */
	public String getPriceAsString() { return String.valueOf(this.film.getPrice()); }
	
	/**
	 * Zwraca ilosc wolnych miejsc jako String.
	 *
	 * @return Ilosc wolnych miejsc jako String.
	 */
	public String getSeatPlanAsString() { return String.valueOf(seatPlan); }
	
}
