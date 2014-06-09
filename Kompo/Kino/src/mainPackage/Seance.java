package mainPackage;

import java.util.Date;

public class Seance {

	public enum Genre {THRILLER, COMEDY, ROMANCE, WAR, CARTOON, UNKNOWN};
	
	private Genre myGenre;
	private String title;
	private Date date;
	private String seatPlan;
	private double price;
	
	public Seance()
	{
		this.myGenre = Genre.UNKNOWN;
		this.title = "none";
		this.date = new Date();
		this.seatPlan = "none";
		this.price = 0.0;
	}
	
	public Seance(Genre myGenre, String title, Date date, String seatPlan, double price)
	{
		this.myGenre = myGenre;
		this.title = title;
		this.date = date;
		this.seatPlan = seatPlan;
		this.price = price;
	}
	
	public Genre getGenre() { return myGenre; }
	public String getTitle() { return title; }
	public Date getDate() { return date; }
	public String getSeatPlan() { return seatPlan; }
	public double getPrice() { return price; }
	
	public String getDateAsString() { return date.toString(); }
	public String getPriceAsString() { return String.valueOf(price); }
	public String getGenreAsString() { return myGenre.toString(); }
}
