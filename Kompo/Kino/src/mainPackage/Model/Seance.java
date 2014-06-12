package mainPackage.Model;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

/**
 * 
 * Reprezentuje jeden, konkretny seans filmowy
 *
 */
public class Seance {
	
	private String myGenre;
	private String title;
	private Date date;
	private int seatPlan;
	private double price;
	
	public Seance()
	{
		this.myGenre = "nieznany";
		this.title = "brak";
		this.date = new Date();
		this.seatPlan = 0;
		this.price = 0.0;
	}
	
	public Seance(String myGenre, String title, Date date, int seatPlan, double price)
	{
		this.myGenre = myGenre;
		this.title = title;
		this.date = date;
		this.seatPlan = seatPlan;
		this.price = price;
	}
	
	public String getGenre() { return myGenre; }
	public String getTitle() { return title; }
	public Date getDate() { return date; }
	public int getSeatPlan() { return seatPlan; }
	public double getPrice() { return price; }
	
	public String getDateAsString() 
	{ 
		Calendar cal = Calendar.getInstance();
		cal.setTime(date);
		String myTime = String.valueOf(cal.get(Calendar.HOUR)) + ":" + String.valueOf(cal.get(Calendar.MINUTE));
		String myDate = String.valueOf(cal.get(Calendar.DAY_OF_MONTH)) + "-" + String.valueOf(cal.get(Calendar.MONTH)) + "-" + String.valueOf(cal.get(Calendar.YEAR));
		String finalDate = myDate + " " + myTime;
		return finalDate; 
	}
	public String getPriceAsString() { return String.valueOf(price); }
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
