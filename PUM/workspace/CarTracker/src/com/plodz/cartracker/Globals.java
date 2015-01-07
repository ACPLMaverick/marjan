package com.plodz.cartracker;

import java.util.GregorianCalendar;

public class Globals {
	public static enum fuelType {ON, PB95, PB98, LPG};
	
	public static fuelType myFuelType;
	public static float myFuelConsumption;
	public static float DBG_updateRatio;
	public static int checkDelay;
	public static float priceON;
	public static float priceLPG;
	public static float pricePB95;
	public static float pricePB98;
	public static GregorianCalendar lastUpdate;
	public static float mapZoomMultiplier;
	
	public static String fuelURL = "http://www.e-petrol.pl/notowania/rynek-krajowy/ceny-stacje-paliw";
	public static String stringON = "ON";
	public static String stringLPG = "LPG";
	public static String stringPB95 = "PB95";
	public static String stringPB98 = "PB98";
}
