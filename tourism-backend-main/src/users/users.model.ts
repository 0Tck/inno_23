import { Column, DataType, Model, Table } from 'sequelize-typescript';

@Table
export class User extends Model<User> {
  @Column({
    type: DataType.INTEGER.UNSIGNED,
    autoIncrement: true,
    primaryKey: true,
  })
  userId: number;

  @Column({ type: DataType.TEXT })
  username: string;

  @Column({ type: DataType.TEXT })
  userEmail: string;

  @Column({ type: DataType.TEXT })
  password: string;
}
