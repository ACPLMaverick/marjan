package mainPackage.Model;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;

/**
 * 
 * Reprezentuje jeden, konkretny seans filmowy
 *
 */
public class Seance {
	
	private Film film;
	private Date date;
	private int seatPlan;
	
	public Seance()
	{
		this.film = new Film();
		this.date = new Date();
		this.seatPlan = 0;
	}
	
	public Seance(Film film, Date date, int seatPlan)
	{
		this.film = film;
		this.date = date;
		this.seatPlan = seatPlan;
	}
	
	public Film getFilm() { return film; }
	public String getGenre() { return this.film.getGenre(); }
	public String getTitle() { return this.film.getTitle(); }
	public Date getDate() { return date; }
	public int getSeatPlan() { return seatPlan; }
	public double getPrice() { return this.film.getPrice(); }
	
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
	public String getPriceAsString() { return String.valueOf(this.film.getPrice()); }
	public String getSeatPlanAsString() { return String.valueOf(seatPlan); }
	
	
	/**
	 *	
	 * @return zwraca iloœæ pól klasy Seans
	 */
	public int getFieldsCount()
	{
		return 5;
	}
}
