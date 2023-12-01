import { Column, DataType, Model, Table } from 'sequelize-typescript';

@Table
export class Sample extends Model<Sample> {
  @Column({
    type: DataType.INTEGER.UNSIGNED,
    autoIncrement: true,
    primaryKey: true,
  })
  id: number;

  @Column({ type: DataType.TEXT })
  columnOne: string;

  @Column({ type: DataType.TEXT })
  columnTwo: string;
}
